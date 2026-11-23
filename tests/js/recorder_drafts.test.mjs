// The pending WAV of the Sound Scout recorder in IndexedDB until the server receipt (#109), against
// a fake stock SDK and an in-memory IndexedDB with just the calls the plugin makes. A reload is a
// fresh load of plugin.js on the same fake database. It proves the contract of the plugin, not how
// the real Hermes Desktop renderer behaves.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { loadPlugin, RECORDER_PATH } from './load_plugin.mjs'

const JOB = 'b'.repeat(32)
const DRAFT_ID = '010203040506'
const REF = '@file:attachments/voice.wav'
const MIB = 1024 * 1024

const same = (actual, expected) => assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected)
const tick = () => new Promise(resolve => setImmediate(resolve))
const named = (calls, name) => calls.filter(call => call[0] === name)
const random = n => Uint8Array.from({ length: n }, (_, i) => i + 1)

// Requests and transactions finish on a later macrotask, as in a browser.
function fakeIndexedDB({ fail = null } = {}) {
  const stores = new Map()
  const later = fn => setImmediate(fn)
  const idb = {
    stores,
    fail,
    opened: 0,
    open(name, version) {
      if (idb.fail === 'throw') throw new Error('private-idb-throw')
      idb.opened++
      const request = {}
      later(() => {
        if (idb.fail === 'open') {
          request.error = new Error('private-idb-open')
          request.onerror?.()
          return
        }
        const fresh = !stores.has(name)
        if (fresh) stores.set(name, new Map())
        const objectStores = stores.get(name)
        const db = {
          version,
          createObjectStore(storeName, { keyPath }) {
            // A real database creates stores only inside its versionchange (upgrade) callback.
            assert.ok(upgrading, 'createObjectStore outside onupgradeneeded')
            objectStores.set(storeName, { keyPath, rows: new Map() })
          },
          transaction(storeName, mode) {
            if (!objectStores.has(storeName)) throw new DOMException(`No store ${storeName}`, 'NotFoundError')
            assert.ok(['readonly', 'readwrite'].includes(mode), `bad mode ${mode}`)
            const tx = {}
            const store = objectStores.get(storeName)
            const ops = []
            // The transaction takes requests only in the task that made it, as a real one does.
            let active = true
            const request = op => {
              if (!active) throw new DOMException('Transaction inactive', 'TransactionInactiveError')
              const req = {}
              ops.push(() => op(req))
              return req
            }
            tx.objectStore = name => {
              assert.equal(name, storeName)
              return {
                put(value) {
                  if (mode !== 'readwrite') throw new DOMException('Read-only transaction', 'ReadOnlyError')
                  const copy = structuredClone(value)
                  if (typeof copy[store.keyPath] !== 'string') throw new DOMException('No key', 'DataError')
                  return request(req => {
                    if (idb.fail === 'put') throw new Error('private-idb-put')
                    store.rows.set(copy[store.keyPath], copy)
                    req.result = copy[store.keyPath]
                  })
                },
                delete(key) {
                  if (mode !== 'readwrite') throw new DOMException('Read-only transaction', 'ReadOnlyError')
                  if (typeof key !== 'string') throw new DOMException('Bad key', 'DataError')
                  return request(() => {
                    if (idb.fail === 'delete') throw new Error('private-idb-delete')
                    store.rows.delete(key)
                  })
                },
                getAll() {
                  return request(req => { req.result = [...store.rows.values()].map(value => structuredClone(value)) })
                }
              }
            }
            later(() => {
              active = false
              try {
                ops.forEach(op => op())
              } catch (error) {
                tx.error = error
                tx.onerror?.()
                return
              }
              tx.oncomplete?.()
            })
            return tx
          }
        }
        request.result = db
        let upgrading = fresh
        if (fresh) request.onupgradeneeded?.()
        upgrading = false
        request.onsuccess?.()
      })
      return request
    },
    rows: () => [...(stores.get('sound-scout-recorder')?.get('drafts')?.rows.values() ?? [])]
  }
  return idb
}

function fakeHost() {
  const local = { connectionId: 'conn-local', mode: 'local', profile: 'default', targetProfile: 'default' }
  const remote = { connectionId: 'conn-remote', mode: 'remote', profile: 'default', targetProfile: 'default' }
  const focus = { stored: 'stored-a', owner: { connectionId: 'conn-remote', profile: 'default' }, connectionId: 'conn-remote', profile: 'default' }
  const calls = []
  const host = {
    state: {
      focusedStoredSessionId: { get: () => focus.stored },
      focusedSessionOwner: { get: () => focus.owner },
      connectionId: { get: () => focus.connectionId },
      profile: { get: () => focus.profile }
    },
    routes: [local, remote],
    refuse: null,
    profileRoutes: async () => {
      calls.push(['profileRoutes'])
      return host.routes
    },
    retainProfile: async route => {
      calls.push(['retain', route])
      return () => calls.push(['release'])
    },
    requestProfile: async (route, method, params) => {
      calls.push([method, structuredClone(params), route])
      if (host.refuse === method) throw new Error(`private-${method}-failure`)
      if (method === 'session.resume') return { session_id: `runtime-${params.session_id}` }
      if (method === 'file.attach') return { attached: true, ref_text: REF }
      if (method === 'prompt.submit') return { admission_id: 'admission-1' }
      throw new Error(`Unexpected RPC: ${method}`)
    },
    notify: value => calls.push(['notify', structuredClone(value)])
  }
  return { host, focus, calls, local, remote }
}

function fakeCtx() {
  const ctx = {
    rests: [],
    registered: [],
    many: [],
    disposers: [],
    health: { ok: true, budgets: { max_duration_seconds: 180, max_input_bytes: 64 * MIB }, intake: { ready: true } },
    post: { ok: true, job_id: JOB },
    async rest(path, opts = {}) {
      ctx.rests.push({ path, opts })
      if (path === '/health') return ctx.health
      if (path.startsWith('/jobs?')) return ctx.post
      return { ok: false, error: 'job_not_found' }
    },
    register: value => ctx.registered.push(value),
    registerMany: values => ctx.many.push(...values),
    onDispose: fn => ctx.disposers.push(fn)
  }
  return ctx
}

function fakeAudio() {
  const capture = { processor: null }
  const stream = {
    getTracks: () => [{ stop() {} }],
    getAudioTracks: () => [{ getSettings: () => ({}) }]
  }
  class AudioContext {
    constructor() { this.sampleRate = 48000 }
    async resume() {}
    async close() {}
    createMediaStreamSource() { return { connect() {}, disconnect() {} } }
    createScriptProcessor() {
      capture.processor = { connect() {}, disconnect() {}, onaudioprocess: null }
      return capture.processor
    }
    get destination() { return {} }
  }
  return { capture, env: { AudioContext, navigator: { mediaDevices: { getUserMedia: async () => stream } } } }
}

function feed(capture) {
  capture.processor.onaudioprocess({
    inputBuffer: { numberOfChannels: 2, getChannelData: channel => new Float32Array(channel ? [0, 0, 0] : [0, 0.5, -0.5]) }
  })
}

// One Desktop window: a plugin load with its own host and backend, and the shared IndexedDB.
async function load(idb, { clock = 1000 } = {}) {
  const { env, capture } = fakeAudio()
  const fake = fakeHost()
  const ctx = fakeCtx()
  const globals = idb === undefined ? env : { ...env, indexedDB: idb }
  const plugin = await loadPlugin({ path: RECORDER_PATH, globals, sdk: { host: fake.host } })
  const recorder = new plugin.Recorder(fake.host, ctx, { sleep: async () => {}, now: () => clock, random })
  return { plugin, recorder, ctx, capture, ...fake }
}

async function record(window) {
  await window.recorder.start()
  feed(window.capture)
  window.recorder.stop()
  await window.recorder.draftWrites
}

async function wavBytes(recorder) {
  return Buffer.from(await recorder.pending.blob.arrayBuffer())
}

function textContent(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node !== 'object') return String(node)
  return [].concat(node.props?.children || []).map(textContent).join('')
}

function findButton(node, label) {
  if (node?.type === 'button' && textContent(node) === label) return node
  return [].concat(node?.props?.children || []).map(child => findButton(child, label)).find(Boolean)
}

test('stop stores the WAV and the chat bound at record start until Send gets the prompt.submit ACK', async () => {
  const idb = fakeIndexedDB()
  const window = await load(idb)
  await record(window)
  const [draft] = idb.rows()
  assert.equal(idb.rows().length, 1)
  same({ ...draft, wav: undefined }, {
    token: DRAFT_ID, createdAt: 1000, target: { stored: 'stored-a', owner: { connectionId: 'conn-remote', profile: 'default' } },
    owner: null, attach: null, submitted: false
  })
  assert.deepEqual(Buffer.from(draft.wav), await wavBytes(window.recorder))
  assert.equal(window.recorder.state.get().unprotected, false)
  await window.recorder.send()
  await window.recorder.draftWrites
  same(idb.rows(), [])
  assert.equal(window.recorder.state.get().phase, 'idle')
})

test('a failed send keeps the draft, and after the attach it keeps the ref_text so a retry does no second attach', async () => {
  const idb = fakeIndexedDB()
  const window = await load(idb)
  await record(window)
  window.host.refuse = 'file.attach'
  await assert.rejects(window.recorder.send())
  await window.recorder.draftWrites
  same(idb.rows().map(row => [row.attach, row.submitted]), [[null, false]])
  window.host.refuse = 'prompt.submit'
  await assert.rejects(window.recorder.send())
  await window.recorder.draftWrites
  const [draft] = idb.rows()
  same([draft.attach, draft.submitted], [{ ref_text: REF, name: `sound-scout-${DRAFT_ID}.wav` }, false])
  same(draft.owner, { connectionId: 'conn-remote', profile: 'default', targetProfile: 'default' })
  window.host.refuse = null
  await window.recorder.send()
  await window.recorder.draftWrites
  same(idb.rows(), [])
  assert.equal(named(window.calls, 'file.attach').length, 2)
})

test('Send and analyze keeps the draft past the prompt.submit ACK until POST /jobs accepts the job', async () => {
  const idb = fakeIndexedDB()
  const window = await load(idb)
  await record(window)
  window.ctx.post = { ok: false, error: 'queue_full' }
  await assert.rejects(window.recorder.send({ analyze: true }), error => error.code === 'queue_full')
  await window.recorder.draftWrites
  same(idb.rows().map(row => [row.attach?.ref_text, row.submitted]), [[REF, true]])
  window.ctx.post = { ok: true, job_id: JOB }
  await window.recorder.send({ analyze: true })
  await window.recorder.draftWrites
  same(idb.rows(), [])
  assert.equal(named(window.calls, 'prompt.submit').length, 1)
  window.recorder.dispose()
})

test('after a reload the plugin offers the pending recording and sends it to the stored chat, not the focused one', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  const recorded = await wavBytes(first.recorder)
  first.recorder.dispose()
  same(idb.rows().map(row => row.token), [DRAFT_ID], 'unloading the plugin keeps the draft')

  const second = await load(idb)
  second.focus.stored = 'stored-b'
  second.focus.owner = { connectionId: 'conn-local', profile: 'default' }
  await second.recorder.restore()
  const state = second.recorder.state.get()
  same([state.phase, state.restored, state.canAnalyze], ['ready', true, true])
  same(named(second.calls, 'notify').map(call => call[1].kind), ['info'])
  assert.deepEqual(await wavBytes(second.recorder), recorded)
  await second.recorder.send()
  same(named(second.calls, 'session.resume').map(call => call[1].session_id), ['stored-a'])
  for (const name of ['retain', 'session.resume', 'file.attach', 'prompt.submit']) {
    assert.equal(named(second.calls, name)[0][name === 'retain' ? 1 : 2], second.remote, name)
  }
  assert.equal(JSON.stringify(second.calls).includes('stored-b'), false)
  same(named(second.calls, 'prompt.submit')[0][1].submission_id, `sound-scout-${DRAFT_ID}`)
  await second.recorder.draftWrites
  same(idb.rows(), [])

  const third = await load(idb)
  await third.recorder.restore()
  assert.equal(third.recorder.state.get().phase, 'idle')
})

test('a draft restored after the attach submits the stored ref_text on the stored route without a second attach', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  first.host.refuse = 'prompt.submit'
  await assert.rejects(first.recorder.send())
  await first.recorder.draftWrites
  first.recorder.dispose()

  const second = await load(idb)
  second.focus.owner = { connectionId: 'conn-local', profile: 'default' }
  await second.recorder.restore()
  await second.recorder.send()
  same(named(second.calls, 'file.attach'), [])
  const [submit] = named(second.calls, 'prompt.submit')
  same(submit[1], { session_id: 'runtime-stored-a', text: REF, submission_id: `sound-scout-${DRAFT_ID}` })
  assert.equal(submit[2], second.remote)
})

test('a restored draft whose stored route is gone is refused, stays stored and is deleted on request', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  first.recorder.dispose()

  const second = await load(idb)
  second.host.routes = [second.local]
  second.focus.owner = { connectionId: 'conn-local', profile: 'default' }
  await second.recorder.restore()
  await assert.rejects(second.recorder.send(), error => error.code === 'ambiguous_route')
  same(second.calls.map(call => call[0]).filter(name => name !== 'notify'), ['profileRoutes'])
  const state = second.recorder.state.get()
  same([state.phase, state.restored], ['ready', true])
  assert.match(state.error, /сервер исходного чата/)
  await second.recorder.draftWrites
  same(idb.rows().map(row => row.token), [DRAFT_ID])
  second.recorder.cancel()
  await second.recorder.draftWrites
  same(idb.rows(), [])
  assert.equal(second.recorder.state.get().phase, 'idle')
})

test('the registered recorder shows the restored draft with Delete, and Delete removes it', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  first.recorder.dispose()

  const { env } = fakeAudio()
  const fake = fakeHost()
  const ctx = fakeCtx()
  const plugin = await loadPlugin({ path: RECORDER_PATH, globals: { ...env, indexedDB: idb }, sdk: { host: fake.host } })
  plugin.default.register(ctx)
  const element = ctx.registered[0].render()
  const render = () => element.type(element.props)
  for (let i = 0; i < 50 && element.props.recorder.state.get().phase !== 'ready'; i++) await tick()
  assert.match(textContent(render()), /Неотправленная запись/)
  assert.ok(findButton(render(), 'Отправить'))
  assert.equal(findButton(render(), 'Отмена'), undefined)
  await findButton(render(), 'Удалить').props.onClick()
  await element.props.recorder.draftWrites
  same(idb.rows(), [])
  assert.ok(findButton(render(), 'Записать'))
  ctx.disposers.forEach(fn => fn())
})

test('Cancel of a fresh recording deletes its draft, and an invalid stored draft is dropped, not offered', async () => {
  const idb = fakeIndexedDB()
  const window = await load(idb)
  await record(window)
  window.recorder.cancel()
  await window.recorder.draftWrites
  same(idb.rows(), [])

  await record(window)
  idb.stores.get('sound-scout-recorder').get('drafts').rows.set('x', { token: 'x', target: { stored: '' } })
  window.recorder.dispose()
  const second = await load(idb)
  await second.recorder.restore()
  await second.recorder.draftWrites
  same(idb.rows().map(row => row.token), [DRAFT_ID])
  assert.equal(second.recorder.state.get().phase, 'ready')
})

test('without a working IndexedDB the recorder still records and sends and says the WAV is not protected', async () => {
  for (const idb of [undefined, fakeIndexedDB({ fail: 'throw' }), fakeIndexedDB({ fail: 'open' }), fakeIndexedDB({ fail: 'put' })]) {
    const window = await load(idb)
    await record(window)
    const state = window.recorder.state.get()
    same([state.phase, state.unprotected, state.error], ['ready', true, ''])
    same(named(window.calls, 'notify'), [])
    await window.recorder.send()
    await window.recorder.draftWrites
    assert.equal(window.recorder.state.get().phase, 'idle')
    assert.equal(named(window.calls, 'prompt.submit').length, 1)
    same(named(window.calls, 'notify').map(call => call[1].kind), ['success'], 'nothing stored, so no warning about a delete')
    await window.recorder.restore()
    assert.equal(window.recorder.state.get().phase, 'idle')
    if (idb) same(idb.rows(), [])
  }
})

test('the registered recorder renders the not-protected note while the WAV waits for a send', async () => {
  const { env, capture } = fakeAudio()
  const fake = fakeHost()
  const ctx = fakeCtx()
  const plugin = await loadPlugin({ path: RECORDER_PATH, globals: { ...env, indexedDB: fakeIndexedDB({ fail: 'open' }) }, sdk: { host: fake.host } })
  plugin.default.register(ctx)
  const element = ctx.registered[0].render()
  const render = () => element.type(element.props)
  const recorder = element.props.recorder
  await recorder.start()
  feed(capture)
  recorder.stop()
  await recorder.draftWrites
  assert.match(textContent(render()), /Запись не защищена от перезагрузки окна/)
  assert.match(textContent(render()), /К отправке/)
  ctx.disposers.forEach(fn => fn())
})

test('after a reload Send and analyze posts the restored draft as a job of the stored chat, not the focused one', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  first.recorder.dispose()

  const second = await load(idb)
  second.focus.stored = 'stored-b'
  second.focus.owner = { connectionId: 'conn-local', profile: 'default' }
  await second.recorder.restore()
  await second.recorder.send({ analyze: true })
  same(named(second.calls, 'session.resume').map(call => [call[1].session_id, call[2].connectionId]), [['stored-a', 'conn-remote']])
  const [post] = second.ctx.rests.filter(call => call.path.startsWith('/jobs?'))
  const query = new URLSearchParams(post.path.slice('/jobs?'.length))
  same([query.get('session_id'), query.get('request_id')], ['stored-a', `sound-scout-desktop-${DRAFT_ID}`])
  assert.equal(JSON.stringify(second.calls).includes('stored-b'), false)
  await second.recorder.draftWrites
  same(idb.rows(), [])
  second.recorder.dispose()
})

test('a restored draft recorded without a known owner is refused while another chat is focused and two routes exist', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  first.focus.owner = null
  await record(first)
  same(idb.rows().map(row => row.target), [{ stored: 'stored-a', owner: null }])
  first.recorder.dispose()

  const second = await load(idb)
  second.focus.stored = 'stored-b'
  await second.recorder.restore()
  await assert.rejects(second.recorder.send(), error => error.code === 'ambiguous_route')
  same(second.calls.map(call => call[0]).filter(name => name !== 'notify'), ['profileRoutes'])
  await second.recorder.draftWrites
  same(idb.rows().map(row => row.token), [DRAFT_ID])
})

test('a failed delete of a stored draft tells the user it comes back on the next load, and it does', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  idb.fail = 'delete'
  first.recorder.cancel()
  await first.recorder.draftWrites
  const warnings = named(first.calls, 'notify').filter(call => call[1].kind === 'warning')
  assert.equal(warnings.length, 1)
  assert.match(warnings[0][1].message, /не удалось удалить.*снова появится при следующей загрузке/)
  same(idb.rows().map(row => row.token), [DRAFT_ID])
  await first.recorder.restore()
  assert.equal(first.recorder.state.get().phase, 'idle', 'the same load does not offer a recording the user deleted')
  first.recorder.dispose()

  idb.fail = null
  const second = await load(idb)
  await second.recorder.restore()
  same([second.recorder.state.get().phase, second.recorder.state.get().restored], ['ready', true])
})

test('a slow /health does not hold back the restored draft: it is offered at once and analysis opens on the answer', async () => {
  const idb = fakeIndexedDB()
  const first = await load(idb)
  await record(first)
  first.recorder.dispose()

  const second = await load(idb)
  let answer
  const rest = second.ctx.rest
  second.ctx.rest = (path, opts) => path === '/health' ? new Promise(resolve => { answer = resolve }) : rest(path, opts)
  const restoring = second.recorder.restore()
  for (let i = 0; i < 50 && !answer; i++) await tick()
  const offered = second.recorder.state.get()
  same([offered.phase, offered.restored, offered.canAnalyze], ['ready', true, false])
  await second.recorder.start()
  assert.equal(second.recorder.pending.token, DRAFT_ID, 'Record does not replace the offered draft')
  answer(second.ctx.health)
  await restoring
  same([second.recorder.state.get().phase, second.recorder.state.get().canAnalyze], ['ready', true])
  await second.recorder.send({ analyze: true })
  await second.recorder.draftWrites
  same(idb.rows(), [])
  second.recorder.dispose()
})
