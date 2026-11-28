// The Sound Scout recorder (#103) against a fake stock SDK, ported from tests/voice-note.test.mjs of
// renidere/hermes-voice-note v0.4.2. It proves the contract of the plugin, not how the real Hermes
// Desktop on Windows behaves: that live check is #105.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { loadPlugin, RECORDER_PATH } from './load_plugin.mjs'

const JOB = 'a'.repeat(32)
const RECORDING_ID = '010203040506'
const REF = '@file:attachments/voice.wav'
const MIB = 1024 * 1024
const FOREGROUND = { spawnPriority: 'foreground' }

// Values built inside the vm context have its own prototypes: compare them as plain JSON.
const same = (actual, expected) => assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected)
const tick = () => new Promise(resolve => setImmediate(resolve))
const named = (calls, name) => calls.filter(call => call[0] === name)
const attached = calls => Buffer.from(named(calls, 'file.attach').at(-1)[1].data_url.split(',')[1], 'base64')
const posted = post => Buffer.from(post.opts.body.data_url.split(',')[1], 'base64')
const IPC_FAILURE = "Error invoking remote method 'hermes:api': Error: File uploads are not supported against OAuth-gated remote backends yet."
const DESKTOP_TEXT = 'File uploads are not supported against OAuth-gated remote backends yet.'
const notices = calls => named(calls, 'notify').map(call => call[1])

function textContent(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node !== 'object') return String(node)
  return [].concat(node.props?.children || []).map(textContent).join('')
}

function findButton(node, label) {
  if (node?.type === 'Button' && textContent(node) === label) return node
  return [].concat(node?.props?.children || []).map(child => findButton(child, label)).find(Boolean)
}

// Calls the function components of a tree, so that the result holds only SDK and DOM elements.
function renderTree(node) {
  if (node == null || typeof node !== 'object') return node
  if (Array.isArray(node)) return node.map(renderTree)
  if (typeof node.type === 'function') return renderTree(node.type(node.props))
  if (node.props?.children === undefined) return node
  return { type: node.type, props: { ...node.props, children: renderTree(node.props.children) } }
}

function* walk(node) {
  if (node == null || typeof node !== 'object') return
  if (Array.isArray(node)) {
    for (const child of node) yield* walk(child)
    return
  }
  yield node
  yield* walk(node.props?.children)
}

const nodes = (tree, type) => [...walk(tree)].filter(node => node.type === type)
const buttonLabels = tree => nodes(tree, 'Button').map(node => node.props['aria-label'])
const stageTexts = tree => nodes(tree, 'li').map(textContent)

function fakeHost(order) {
  const local = { connectionId: 'conn-local', mode: 'local', profile: 'default', targetProfile: 'default' }
  const remote = { connectionId: 'conn-remote', mode: 'remote', profile: 'default', targetProfile: 'default' }
  const focus = {
    stored: 'stored-a',
    owner: { connectionId: 'conn-remote', profile: 'default' },
    connectionId: 'conn-remote',
    profile: 'default'
  }
  const calls = []
  const host = {
    state: {
      focusedStoredSessionId: { get: () => focus.stored },
      focusedSessionOwner: { get: () => focus.owner },
      connectionId: { get: () => focus.connectionId },
      profile: { get: () => focus.profile }
    },
    routes: [local, remote],
    // stored chat → its running turn in the session.resume reply
    turns: {},
    interruptFailure: null,
    profileRoutes: async () => {
      calls.push(['profileRoutes'])
      return host.routes
    },
    retainProfile: async (route, options) => {
      calls.push(['retain', route, structuredClone(options)])
      return () => calls.push(['release'])
    },
    requestProfile: async (route, method, params, timeoutMs, options) => {
      calls.push([method, structuredClone(params), route, timeoutMs, options && structuredClone(options)])
      order.push(method)
      if (method === 'session.resume') {
        return { session_id: params.session_id.replace('stored-', 'runtime-'), stored_session_id: params.session_id, ...host.turns[params.session_id] }
      }
      if (method === 'session.interrupt') {
        if (host.interruptFailure) throw host.interruptFailure
        return { execution_state: 'idle' }
      }
      if (method === 'file.attach') return { attached: true, ref_text: REF }
      if (method === 'prompt.submit') return { admission_id: 'admission-1', status: 'accepted' }
      throw new Error(`Unexpected RPC: ${method}`)
    },
    retainSession: () => {
      calls.push(['retainSession'])
      throw new Error('retainSession is not a stock SDK call')
    },
    composer: {
      submit: () => {
        calls.push(['composer.submit'])
        throw new Error('composer.submit is not the delivery path')
      }
    },
    notify: value => calls.push(['notify', structuredClone(value)])
  }
  return { host, focus, calls, local, remote }
}

function fakeCtx(order) {
  const ctx = {
    rests: [],
    registered: [],
    many: [],
    disposers: [],
    health: {
      ok: true,
      budgets: { max_duration_seconds: 180, max_input_bytes: 64 * MIB },
      intake: { ready: true, error: null, detail: null }
    },
    post: { ok: true, job_id: JOB, request_id: `sound-scout-desktop-${RECORDING_ID}`, state: 'queued', reused: false },
    statuses: [],
    actions: [],
    async rest(path, opts = {}) {
      ctx.rests.push({ path, opts })
      order.push(`rest ${opts.method || 'GET'} ${path.split('?')[0]}`)
      const answer = path === '/health' ? ctx.health
        : path.startsWith('/jobs?') ? ctx.post
          : /^\/jobs\/[^/]+\/(report|cancel|retry)$/.test(path) ? (ctx.actions.shift() ?? { ok: false, error: 'job_not_found' })
            : path.startsWith('/jobs/') ? (ctx.statuses.shift() ?? { ok: false, error: 'job_not_found' })
              : new Error(`Unexpected rest path: ${path}`)
      if (answer instanceof Error) throw answer
      return typeof answer === 'function' ? answer() : answer
    },
    register: value => ctx.registered.push(value),
    registerMany: values => ctx.many.push(...values),
    onDispose: fn => ctx.disposers.push(fn)
  }
  return ctx
}

function fakeAudio() {
  const capture = { tracksStopped: 0, contextClosed: 0, constraints: null, processor: null }
  const stream = {
    getTracks: () => [{ stop() { capture.tracksStopped++ } }],
    getAudioTracks: () => [{ getSettings: () => ({ channelCount: 2, echoCancellation: false, noiseSuppression: false, autoGainControl: false }) }]
  }
  class AudioContext {
    constructor() { this.sampleRate = 48000; this.state = 'running' }
    async resume() {}
    async close() { this.state = 'closed'; capture.contextClosed++ }
    createMediaStreamSource() { return { connect() {}, disconnect() {} } }
    createScriptProcessor() {
      capture.processor = { connect() {}, disconnect() {}, onaudioprocess: null }
      return capture.processor
    }
    get destination() { return {} }
  }
  const env = {
    AudioContext,
    navigator: {
      mediaDevices: {
        getUserMedia: async value => {
          capture.constraints = structuredClone(value)
          return stream
        }
      }
    }
  }
  return { capture, env }
}

function feed(capture) {
  capture.processor.onaudioprocess({
    inputBuffer: { numberOfChannels: 2, getChannelData: channel => new Float32Array(channel ? [0, 0, 0] : [0, 0.5, -0.5]) }
  })
}

function feedFrames(capture, frames) {
  capture.processor.onaudioprocess({
    inputBuffer: { numberOfChannels: 2, getChannelData: () => new Float32Array(frames).fill(0.25) }
  })
}

const random = n => Uint8Array.from({ length: n }, (_, i) => i + 1)

// The composer the card renders in; its storedSessionId is the chat of the composer.
function fakeComposer() {
  const composer = { stored: 'stored-a' }
  return { composer, sdk: { useComposerSession: () => ({ sessionId: 'runtime-a', storedSessionId: composer.stored }) } }
}

async function setup({ globals = {}, options = {} } = {}) {
  const order = []
  const { env, capture } = fakeAudio()
  const { composer, sdk } = fakeComposer()
  const plugin = await loadPlugin({ path: RECORDER_PATH, sdk, globals: { ...env, ...globals } })
  const fake = fakeHost(order)
  const ctx = fakeCtx(order)
  const sleeps = []
  let clock = 0
  const recorder = new plugin.Recorder(fake.host, ctx, {
    sleep: async ms => { sleeps.push(ms); clock += ms },
    now: () => clock,
    random,
    ...options
  })
  const card = () => renderTree({ type: plugin.JobCard, props: { recorder } })
  const press = label => findButton(card(), label).props.onClick()
  return { plugin, ...fake, ctx, recorder, env, capture, sleeps, order, composer, card, press }
}

async function setupRegistered({ globals = {}, host: patch, sdk: extra = {} } = {}) {
  const order = []
  const { env, capture } = fakeAudio()
  const fake = fakeHost(order)
  patch?.(fake.host)
  const { composer, sdk } = fakeComposer()
  const plugin = await loadPlugin({ path: RECORDER_PATH, sdk: { host: fake.host, ...sdk, ...extra }, globals: { ...env, ...globals } })
  const ctx = fakeCtx(order)
  plugin.default.register(ctx)
  const area = name => ctx.registered.find(item => item.area === name)
  const element = area('composer.actions')?.render()
  const render = () => renderTree(element)
  const card = () => renderTree(area('composer.top').render())
  const click = label => findButton(render(), label).props.onClick()
  return { plugin, ...fake, ctx, env, capture, element, render, card, click, composer, recorder: element?.props.recorder, order }
}

test('encodeWav writes the captured left and silent right channels as PCM16 without downmix', async () => {
  const plugin = await loadPlugin({ path: RECORDER_PATH })
  const blob = plugin.encodeWav([[new Float32Array([0, 0.5, -1]), new Float32Array([0, 0, 0])]], 48000)
  assert.equal(blob.type, 'audio/wav')
  const buffer = await blob.arrayBuffer()
  const view = new DataView(buffer)
  assert.equal(new TextDecoder().decode(buffer.slice(0, 4)), 'RIFF')
  assert.equal(new TextDecoder().decode(buffer.slice(8, 12)), 'WAVE')
  assert.equal(view.getUint16(20, true), 1)
  assert.equal(view.getUint16(22, true), 2)
  assert.equal(view.getUint32(24, true), 48000)
  assert.equal(view.getUint16(34, true), 16)
  assert.equal(view.getUint32(40, true), 12)
  assert.deepEqual([...new Int16Array(buffer, 44)], [0, 0, 16384, 0, -32768, 0])
})

test('Send attaches the WAV and submits its reference with prompt.submit on the captured owner route', async () => {
  const { recorder, calls, capture, remote, ctx } = await setup()
  await recorder.start()
  assert.equal(recorder.state.get().phase, 'recording')
  same(capture.constraints.audio, { channelCount: { ideal: 2 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false })
  same(calls, [])
  feed(capture)
  await recorder.send()
  same(calls.map(call => call[0]), ['profileRoutes', 'retain', 'session.resume', 'file.attach', 'prompt.submit', 'release', 'notify'])
  assert.equal(calls[1][1], remote, 'the route is chosen by connectionId and profile out of two default routes')
  same(calls[1][2], FOREGROUND)
  const [, resume, resumeRoute, , resumeOptions] = calls[2]
  same(resume, { session_id: 'stored-a', omit_messages: true })
  assert.equal(resumeRoute, remote)
  same(resumeOptions, FOREGROUND)
  const [, attach, attachRoute, attachTimeout] = calls[3]
  assert.equal(attachRoute, remote)
  assert.equal(attachTimeout, 120000)
  assert.equal(attach.session_id, 'runtime-a')
  assert.equal(attach.name, `sound-scout-${RECORDING_ID}.wav`)
  assert.match(attach.data_url, /^data:audio\/wav;base64,UklGR/)
  same(calls[4][1], { session_id: 'runtime-a', text: REF, submission_id: `sound-scout-${RECORDING_ID}` })
  assert.equal(calls[4][2], remote)
  same(calls[6][1], { kind: 'success', message: 'Голосовое отправлено' })
  assert.equal(recorder.state.get().phase, 'idle')
  assert.equal(recorder.pending, null)
  assert.equal(capture.tracksStopped, 1)
  assert.equal(capture.contextClosed, 1)
  same(ctx.rests.map(r => r.path), ['/health'])
})

test('a focus change, a rename or a new owner after the start do not move the send to another chat', async () => {
  for (const change of [
    focus => { focus.stored = 'stored-b'; focus.owner = { connectionId: 'conn-local', profile: 'default' } },
    focus => { focus.owner.connectionId = 'conn-local'; focus.owner.profile = 'other' },
    focus => { focus.owner = null }
  ]) {
    const { recorder, calls, capture, focus, remote } = await setup()
    await recorder.start()
    feed(capture)
    change(focus)
    await recorder.send()
    same(named(calls, 'session.resume')[0][1], { session_id: 'stored-a', omit_messages: true })
    for (const name of ['retain', 'session.resume', 'file.attach', 'prompt.submit']) {
      assert.equal(named(calls, name)[0][name === 'retain' ? 1 : 2], remote, name)
    }
    assert.equal(recorder.state.get().phase, 'idle')
  }
})

test('the chat and owner captured at record start receive the voice and the job binding after a focus change', async () => {
  const { recorder, calls, capture, focus, remote, ctx } = await setup()
  await recorder.start()
  feed(capture)
  focus.stored = 'stored-b'
  focus.owner = { connectionId: 'conn-remote', profile: 'other' }
  await recorder.send({ analyze: true })
  same(named(calls, 'session.resume').map(call => call[1].session_id), ['stored-a'])
  assert.equal(named(calls, 'prompt.submit')[0][2], remote)
  const posts = ctx.rests.filter(r => r.opts.method === 'POST')
  assert.equal(posts.length, 1)
  const query = new URL(`http://plugin${posts[0].path}`).searchParams
  assert.equal(query.get('session_id'), 'stored-a')
  assert.equal(query.get('profile'), 'default')
  assert.equal(JSON.stringify(calls).includes('stored-b'), false)
  assert.equal(posts[0].path.includes('stored-b'), false)
})

test('a null owner is read again only while the chat keeps focus, and two routes refuse ambiguous_route', async () => {
  {
    const { recorder, calls, capture, focus, remote, local } = await setup()
    focus.owner = null
    await recorder.start()
    same(calls, [], 'no route lookup while recording')
    feed(capture)
    focus.owner = { connectionId: 'conn-local', profile: 'default' }
    await recorder.send()
    assert.equal(named(calls, 'prompt.submit')[0][2], local)
    assert.notEqual(local, remote)
  }
  {
    const { recorder, calls, capture, focus } = await setup()
    focus.owner = null
    await recorder.start()
    feed(capture)
    focus.stored = 'stored-b'
    focus.owner = { connectionId: 'conn-remote', profile: 'default' }
    await assert.rejects(recorder.send(), error => error.code === 'ambiguous_route')
    assert.equal(recorder.state.get().phase, 'ready')
    assert.match(recorder.state.get().error, /однозначно определить сервер исходного чата/)
    same(calls.map(call => call[0]), ['profileRoutes', 'notify'])
  }
  {
    const { recorder, calls, capture, focus, host, remote } = await setup()
    focus.owner = null
    host.routes = [remote]
    await recorder.start()
    feed(capture)
    await recorder.send()
    assert.equal(named(calls, 'prompt.submit')[0][2], remote)
    same(named(calls, 'prompt.submit')[0][1].session_id, 'runtime-a')
  }
})

test('an unsaved draft chat is refused before the microphone and the health call', async () => {
  const { recorder, focus, capture, ctx, calls } = await setup()
  focus.stored = null
  await recorder.start()
  assert.equal(capture.constraints, null)
  assert.equal(recorder.state.get().phase, 'idle')
  assert.match(recorder.state.get().error, /существующий чат/)
  same(ctx.rests, [])
  assert.equal(notices(calls).length, 1)
})

test('a failure before the attach keeps the WAV, names the step and a retry uploads the same bytes', async () => {
  const cases = [
    { setup: host => { host.profileRoutes = async () => { throw new Error('private-route-failure') } }, message: /подключить исходный чат/ },
    { setup: host => { host.retainProfile = async () => { throw new Error('private-connect-failure') } }, message: /подключить исходный чат/ },
    { setup: host => { host.requestProfile = async () => { throw new Error('private-resume-failure') } }, message: /открыть исходный чат/ },
    {
      setup: host => {
        const request = host.requestProfile
        host.requestProfile = (...args) => args[1] === 'file.attach' ? Promise.reject(new Error('private-attach-failure')) : request(...args)
      },
      message: /загрузить аудио/
    }
  ]
  for (const scenario of cases) {
    const { recorder, host, calls, capture } = await setup()
    const healthy = { ...host }
    scenario.setup(host)
    await recorder.start()
    feed(capture)
    await assert.rejects(recorder.send(), /private-/)
    const recorded = Buffer.from(await recorder.pending.blob.arrayBuffer())
    assert.equal(recorder.state.get().phase, 'ready')
    assert.match(recorder.state.get().error, scenario.message)
    assert.match(recorder.state.get().error, /Запись сохранена/)
    assert.doesNotMatch(recorder.state.get().error, /private-/)
    assert.equal(named(calls, 'prompt.submit').length, 0)
    assert.equal(capture.tracksStopped, 1)
    Object.assign(host, healthy)
    await recorder.send()
    assert.deepEqual(attached(calls), recorded)
    assert.equal(named(calls, 'prompt.submit').length, 1)
    assert.equal(named(calls, 'release').length, named(calls, 'retain').length)
    assert.equal(recorder.state.get().phase, 'idle')
  }
})

test('a failed prompt.submit retries without a second file.attach, with the same ref_text and submission_id', async () => {
  for (const refusal of [() => Promise.reject(new Error('private-submit-failure')), () => Promise.resolve({ status: 'refused' })]) {
    const { recorder, host, calls, capture } = await setup()
    const request = host.requestProfile
    let refuse = true
    host.requestProfile = (...args) => {
      if (args[1] === 'prompt.submit' && refuse) {
        calls.push(['prompt.submit', structuredClone(args[2])])
        return refusal()
      }
      return request(...args)
    }
    await recorder.start()
    feed(capture)
    await assert.rejects(recorder.send())
    assert.equal(recorder.state.get().phase, 'ready')
    assert.match(recorder.state.get().error, /Чат не принял сообщение/)
    assert.equal(named(calls, 'release').length, 1)
    refuse = false
    await recorder.send()
    assert.equal(named(calls, 'file.attach').length, 1)
    const submits = named(calls, 'prompt.submit')
    assert.equal(submits.length, 2)
    same(submits[0][1], { session_id: 'runtime-a', text: REF, submission_id: `sound-scout-${RECORDING_ID}` })
    same(submits[1][1], submits[0][1])
    assert.equal(named(calls, 'session.resume').length, 2)
    assert.equal(named(calls, 'retain').length, 2)
    assert.equal(named(calls, 'release').length, 2)
    assert.equal(recorder.state.get().phase, 'idle')
  }
})

test('cancelling an in-flight attach never submits the late reply nor posts a job', async () => {
  const { recorder, host, calls, capture, ctx } = await setup()
  const request = host.requestProfile
  let finish
  host.requestProfile = (...args) => args[1] === 'file.attach' ? new Promise(resolve => { finish = resolve }) : request(...args)
  await recorder.start()
  feed(capture)
  const sending = recorder.send({ analyze: true })
  await tick()
  recorder.cancel()
  finish({ attached: true, ref_text: '@file:attachments/cancelled.wav' })
  await sending
  assert.equal(named(calls, 'prompt.submit').length, 0)
  assert.equal(named(calls, 'release').length, 1)
  assert.equal(ctx.rests.filter(r => r.path.startsWith('/jobs')).length, 0)
  assert.equal(recorder.state.get().phase, 'idle')
  assert.equal(recorder.pending, null)
})

test('cancelling while the microphone permission is pending closes the late stream', async () => {
  const { recorder, calls, capture, env } = await setup()
  let grant
  env.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { grant = resolve })
  const starting = recorder.start()
  await tick()
  recorder.cancel()
  grant({ getTracks: () => [{ stop() { capture.tracksStopped++ } }] })
  await starting
  assert.equal(capture.tracksStopped, 1)
  assert.equal(capture.processor, null)
  assert.equal(recorder.state.get().phase, 'idle')
  same(calls, [])
})

test('a cancelled route lease is released late without touching a newer recording', async () => {
  const { recorder, host, calls, capture } = await setup()
  let finish
  host.retainProfile = () => new Promise(resolve => { finish = resolve })
  await recorder.start()
  feed(capture)
  const sending = recorder.send()
  await tick()
  recorder.cancel()
  await recorder.start()
  feed(capture)
  finish(() => calls.push(['release']))
  await sending
  assert.equal(recorder.state.get().phase, 'recording')
  assert.ok(recorder.chunks.length)
  same(calls.filter(call => call[0] !== 'profileRoutes'), [['release']])
  recorder.cancel()
})

test('a second click while sending does not start a second upload', async () => {
  const { recorder, host, calls, capture } = await setup()
  const request = host.requestProfile
  const uploads = []
  host.requestProfile = (...args) => args[1] === 'file.attach' ? new Promise(resolve => uploads.push(resolve)) : request(...args)
  await recorder.start()
  feed(capture)
  const first = recorder.send()
  const second = recorder.send({ analyze: true })
  await tick()
  try {
    assert.equal(uploads.length, 1)
  } finally {
    for (const resolve of uploads) resolve({ attached: true, ref_text: REF })
    await Promise.allSettled([first, second])
  }
  assert.equal(named(calls, 'prompt.submit').length, 1)
})

test('capture failures name the microphone step without a server or a raw error', async () => {
  const cases = [
    { name: 'NotAllowedError', message: /нет доступа к микрофону/i },
    { name: 'NotFoundError', message: /микрофон не найден/i },
    { name: 'NotReadableError', message: /микрофон занят или недоступен/i },
    { dsp: true, message: /не отключил автоматическую обработку/i },
    { context: true, message: /настроить запись с микрофона/i }
  ]
  for (const scenario of cases) {
    const { recorder, calls, env, capture } = await setup()
    if (scenario.name) {
      env.navigator.mediaDevices.getUserMedia = async () => {
        const error = new Error('private-device-detail')
        error.name = scenario.name
        throw error
      }
    }
    if (scenario.dsp) {
      env.navigator.mediaDevices.getUserMedia = async () => ({
        getTracks: () => [{ stop() { capture.tracksStopped++ } }],
        getAudioTracks: () => [{ getSettings: () => ({ autoGainControl: true }) }]
      })
    }
    if (scenario.context) env.AudioContext.prototype.createMediaStreamSource = () => { throw new Error('private-device-detail') }
    await assert.rejects(recorder.start())
    assert.equal(recorder.state.get().phase, 'idle')
    assert.match(recorder.state.get().error, scenario.message)
    assert.doesNotMatch(recorder.state.get().error, /сервер|private-/i)
    same(calls, [['notify', { kind: 'error', message: recorder.state.get().error }]])
    if (scenario.dsp) {
      assert.equal(capture.tracksStopped, 1)
      assert.equal(capture.processor, null)
    }
    if (scenario.context) {
      assert.equal(capture.tracksStopped, 1)
      assert.equal(capture.contextClosed, 1)
    }
  }
})

test('sending before the first audio frame stops the capture instead of a stuck recorder', async () => {
  const { recorder, calls, capture } = await setup()
  await recorder.start()
  await assert.rejects(recorder.send())
  assert.equal(recorder.state.get().phase, 'idle')
  assert.equal(capture.tracksStopped, 1)
  same(calls, [['notify', { kind: 'error', message: recorder.state.get().error }]])
})

test('a queued callback of a cancelled capture does not enter the next recording', async () => {
  const { recorder, capture } = await setup()
  await recorder.start()
  const oldCallback = capture.processor.onaudioprocess
  recorder.cancel()
  await recorder.start()
  oldCallback({ inputBuffer: { numberOfChannels: 2, getChannelData: () => new Float32Array([0.25]) } })
  assert.equal(recorder.chunks.length, 0)
  recorder.cancel()
})

test('an invalid attach reply never becomes the message text', async () => {
  for (const reply of [{ attached: true, ref_text: 'unexpected server text' }, { attached: false, ref_text: REF }, null]) {
    const { recorder, host, calls, capture } = await setup()
    const request = host.requestProfile
    host.requestProfile = (...args) => args[1] === 'file.attach' ? Promise.resolve(reply) : request(...args)
    await recorder.start()
    feed(capture)
    await assert.rejects(recorder.send())
    assert.equal(named(calls, 'prompt.submit').length, 0)
    assert.equal(recorder.state.get().phase, 'ready')
    assert.equal(recorder.pending.attach, null)
  }
})

test('the recording limit is the duration budget of GET /health and keeps the WAV without a send', async () => {
  for (const seconds of [180, 42]) {
    let timeout
    const { recorder, calls, capture, ctx } = await setup({
      globals: { setTimeout: (fn, ms) => { timeout = { fn, ms }; return 1 }, clearTimeout: () => {} }
    })
    ctx.health.budgets.max_duration_seconds = seconds
    await recorder.start()
    feed(capture)
    same(ctx.rests[0], { path: '/health', opts: { timeoutMs: 10000 } })
    assert.equal(timeout.ms, seconds * 1000)
    assert.equal(recorder.state.get().limitSeconds, seconds)
    timeout.fn()
    assert.equal(recorder.state.get().phase, 'ready')
    assert.ok(recorder.pending.blob.size > 44)
    assert.equal(capture.tracksStopped, 1)
    assert.match(recorder.state.get().error, new RegExp(`лимит записи ${seconds} с`))
    same(calls, [['notify', { kind: 'error', message: recorder.state.get().error }]])
    assert.equal(ctx.rests.length, 1)
  }
})

test('the duration budget caps the captured frames, so the WAV is never longer than max_duration_seconds', async () => {
  for (const frames of [[40000, 10000], [48000]]) {
    let timeout
    const { recorder, capture, ctx, calls } = await setup({
      globals: { setTimeout: (fn, ms) => { timeout = { fn, ms }; return 1 }, clearTimeout: () => {} }
    })
    ctx.health.budgets.max_duration_seconds = 1
    await recorder.start()
    for (const [index, count] of frames.entries()) {
      assert.equal(recorder.state.get().phase, 'recording', `before chunk ${index}`)
      feedFrames(capture, count)
    }
    assert.equal(recorder.state.get().phase, 'ready')
    const view = new DataView(await recorder.pending.blob.arrayBuffer())
    assert.equal(view.getUint32(40, true), 48000 * 2 * 2)
    assert.match(recorder.state.get().error, /лимит записи 1 с/)
    assert.equal(timeout.ms, 1000)
    timeout.fn()
    assert.equal(recorder.state.get().phase, 'ready')
    assert.equal(notices(calls).length, 1)
    assert.equal(capture.tracksStopped, 1)
  }
})

test('the byte budget of GET /health stops the capture before the WAV grows past it', async () => {
  const { recorder, calls, capture, ctx } = await setup()
  ctx.health.budgets.max_input_bytes = 44 + 2 * 12
  await recorder.start()
  feed(capture)
  feed(capture)
  assert.equal(recorder.state.get().phase, 'recording')
  feed(capture)
  assert.equal(recorder.state.get().phase, 'ready')
  assert.equal(recorder.pending.blob.size, 68)
  assert.equal(named(calls, 'file.attach').length, 0)
})

const R02 = { max_duration_seconds: 180, max_input_bytes: 104857600 }
const noBudgetWarning = text => `Sound Scout не сообщил лимит записи: ${text}. Запись идёт с лимитом 180 с, разбор недоступен.`

// Without budgets from GET /health the capture starts with the R02 limit and only Send is open (#174).
async function startWithoutBudget(health, patch = () => {}) {
  let timeout
  const context = await setup({ globals: { setTimeout: (fn, ms) => { timeout = { fn, ms }; return 1 }, clearTimeout: () => {} } })
  context.ctx.health = health
  patch(context.ctx)
  await context.recorder.start()
  const { recorder, capture, calls } = context
  assert.notEqual(capture.constraints, null)
  assert.equal(recorder.state.get().phase, 'recording')
  assert.equal(timeout.ms, 180000)
  assert.equal(recorder.state.get().limitSeconds, 180)
  same(recorder.budget, R02)
  assert.equal(recorder.state.get().canAnalyze, false)
  assert.equal(recorder.state.get().error, '')
  const warnings = notices(calls)
  assert.equal(warnings.length, 1)
  assert.equal(warnings[0].kind, 'warning')
  return { ...context, warning: warnings[0].message }
}

test('a GET /health exception starts the recording with the R02 limit and disables Send and analyze', async () => {
  const { recorder, warning } = await startWithoutBudget(new Error(IPC_FAILURE))
  assert.equal(recorder.state.get().intakeError, 'no_answer')
  assert.ok(warning.startsWith(`Sound Scout не сообщил лимит записи: ${DESKTOP_TEXT}.`))
  assert.equal(warning, noBudgetWarning(DESKTOP_TEXT))
})

// ctx.rest of the Desktop SDK rejects when its timeoutMs runs out; the fake rejects the same way.
test('a GET /health timeout of ctx.rest starts the recording with the R02 limit and disables Send and analyze', async () => {
  const timedOut = 'Request timed out after 10000 ms'
  const { recorder, ctx, warning } = await startWithoutBudget(null, ctx => {
    ctx.rest = async (path, opts = {}) => {
      ctx.rests.push({ path, opts })
      if (path === '/health' && opts.timeoutMs === 10000) throw new Error(timedOut)
      throw new Error(`Unexpected rest: ${path} ${JSON.stringify(opts)}`)
    }
  })
  same(ctx.rests, [{ path: '/health', opts: { timeoutMs: 10000 } }])
  assert.equal(recorder.state.get().intakeError, 'no_answer')
  assert.equal(warning, noBudgetWarning(timedOut))
})

test('a GET /health with ok: false starts the recording with the R02 limit and disables Send and analyze', async () => {
  const { recorder, warning } = await startWithoutBudget({ ok: false, error: 'cli_failed', detail: 'private-detail' })
  assert.equal(recorder.state.get().intakeError, 'cli_failed')
  assert.equal(warning, noBudgetWarning('cli_failed'))
  assert.doesNotMatch(warning, /private-/)
})

test('a GET /health with null or invalid budgets starts the recording with the R02 limit and disables Send and analyze', async () => {
  for (const [health, code] of [
    [{ ok: true, budgets: null, intake: { ready: false, error: 'cli_failed', detail: 'private-detail' } }, 'cli_failed'],
    [{ ok: true, budgets: { max_duration_seconds: 0, max_input_bytes: 10 }, intake: { ready: true } }, 'no_budgets'],
    [{ ok: true, budgets: { max_duration_seconds: 1.5, max_input_bytes: 10 }, intake: { ready: true } }, 'no_budgets']
  ]) {
    const { recorder, warning } = await startWithoutBudget(health)
    assert.equal(recorder.state.get().intakeError, code)
    assert.equal(warning, noBudgetWarning(code))
    assert.doesNotMatch(warning, /private-/)
  }
})

test('without budgets Send goes to the chat captured at start by the stock path and Send and analyze stays disabled', async () => {
  const { ctx, click, render, calls, capture, recorder, focus, remote } = await setupRegistered()
  ctx.health = new Error(IPC_FAILURE)
  try {
    await click('Записать')
    assert.equal(recorder.state.get().phase, 'recording')
    feed(capture)
    const analyze = findButton(render(), 'Отправить и разобрать')
    assert.equal(analyze.props.disabled, true)
    assert.match(analyze.props.title, /no_answer/)
    focus.stored = 'stored-b'
    focus.owner = { connectionId: 'conn-local', profile: 'default' }
    await click('Отправить')
    same(calls.map(call => call[0]).filter(name => name !== 'notify'),
      ['profileRoutes', 'retain', 'session.resume', 'file.attach', 'prompt.submit', 'release'])
    same(named(calls, 'session.resume')[0][1], { session_id: 'stored-a', omit_messages: true })
    assert.equal(named(calls, 'prompt.submit')[0][2], remote)
    assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 0)
    same(ctx.rests.map(r => r.path), ['/health'])
    assert.equal(recorder.state.get().phase, 'idle')
  } finally {
    ctx.disposers.forEach(fn => fn())
  }
})

const status = (state, extra = {}) => ({ ok: true, job: { job_id: JOB, state, stages: ['decode', 'asr', 'report'], finished_stages: [], current_stage: null, state_reason: null, ...extra }, result: null })

test('Send and analyze posts the same WAV to POST /jobs after prompt.submit and shows the progress by polling', async () => {
  const { recorder, calls, capture, ctx, order, sleeps } = await setup()
  ctx.statuses = [
    status('queued'),
    status('running'),
    status('running', { finished_stages: [{ stage: 'decode' }], current_stage: { stage: 'asr', started_at: 't' } }),
    { ...status('finished'), result: { job_state: 'finished', status: 'partial' } }
  ]
  const texts = []
  recorder.job.subscribe(job => job && texts.push(job.text))
  await recorder.start()
  feed(capture)
  await recorder.send({ analyze: true })
  same(order.slice(0, 5), ['rest GET /health', 'session.resume', 'file.attach', 'prompt.submit', 'rest POST /jobs'])
  const post = ctx.rests.find(r => r.opts.method === 'POST')
  const query = new URL(`http://plugin${post.path}`).searchParams
  same(Object.fromEntries(query), { request_id: `sound-scout-desktop-${RECORDING_ID}`, session_id: 'stored-a', profile: 'default' })
  same(Object.keys(post.opts).sort(), ['body', 'method', 'timeoutMs'])
  same(Object.keys(post.opts.body), ['data_url'])
  assert.match(post.opts.body.data_url, /^data:audio\/wav;base64,UklGR/)
  assert.equal(post.opts.timeoutMs, 120000)
  assert.deepEqual(posted(post), attached(calls))
  assert.equal(recorder.state.get().phase, 'idle')
  assert.deepEqual(notices(calls)[0], { kind: 'success', message: 'Голосовое отправлено, разбор принят' })
  await recorder.polling
  same(ctx.rests.filter(r => !r.opts.method && r.path !== '/health').map(r => r.path), Array(4).fill(`/jobs/${JOB}`))
  same(texts, [
    'Разбор: принят',
    'Разбор: в очереди',
    'Разбор: идёт',
    'Разбор: этап 2 из 3 — asr',
    'Разбор готов: partial. Итог придёт в чат заметкой'
  ])
  assert.equal(recorder.job.get().final, true)
  same(sleeps, [3000, 3000, 3000, 3000])
  same(notices(calls).at(-1), { kind: 'success', message: 'Разбор готов: partial. Итог придёт в чат заметкой' })
})

test('a refused POST /jobs keeps the voice in the chat and a retry posts again without attach or submit', async () => {
  const { recorder, calls, capture, ctx } = await setup()
  ctx.post = { ok: false, error: 'queue_full', detail: 'private-detail' }
  ctx.statuses = [{ ...status('failed', { state_reason: 'input_invalid' }) }]
  await recorder.start()
  feed(capture)
  await assert.rejects(recorder.send({ analyze: true }), error => error.code === 'queue_full')
  assert.equal(recorder.state.get().phase, 'ready')
  assert.match(recorder.state.get().error, /Голосовое в чате, но разбор не принят: queue_full/)
  assert.doesNotMatch(recorder.state.get().error, /private-/)
  ctx.post = { ok: true, job_id: JOB, reused: false }
  await recorder.send({ analyze: true })
  assert.equal(named(calls, 'file.attach').length, 1)
  assert.equal(named(calls, 'prompt.submit').length, 1)
  const posts = ctx.rests.filter(r => r.opts.method === 'POST')
  assert.equal(posts.length, 2)
  assert.equal(posts[1].path, posts[0].path)
  assert.equal(posts[1].opts.body.data_url, posts[0].opts.body.data_url)
  assert.equal(posts[1].opts.upload, undefined)
  await recorder.polling
  const job = recorder.job.get()
  same([job.jobId, job.text, job.final], [JOB, 'Разбор не удался: input_invalid', true])
})

test('a failed ctx.rest of POST /jobs shows the Desktop error text without the IPC wrapper', async () => {
  const { recorder, calls, capture, ctx } = await setup()
  ctx.post = new Error(IPC_FAILURE)
  await recorder.start()
  feed(capture)
  await assert.rejects(recorder.send({ analyze: true }), error => error.code === 'no_answer' && error.message === DESKTOP_TEXT)
  assert.equal(recorder.state.get().phase, 'ready')
  assert.equal(recorder.state.get().error,
    `Голосовое в чате, но разбор не принят: no_answer — ${DESKTOP_TEXT}. Повтори «Отправить и разобрать».`)
  assert.equal(named(calls, 'file.attach').length, 1)
  ctx.post = { ok: true, job_id: JOB, reused: false }
  await recorder.send({ analyze: true })
  assert.equal(recorder.state.get().phase, 'idle')
  assert.equal(named(calls, 'file.attach').length, 1)
  const posts = ctx.rests.filter(r => r.opts.method === 'POST')
  assert.equal(posts[1].opts.body.data_url, posts[0].opts.body.data_url)
  assert.deepEqual(posted(posts[1]), attached(calls))
})

test('a failed ctx.rest of GET /health and of the last status poll show the Desktop error text', async () => {
  {
    const { recorder, capture, ctx, calls } = await setup()
    ctx.health = new Error(IPC_FAILURE)
    await recorder.start()
    assert.notEqual(capture.constraints, null)
    assert.equal(recorder.state.get().phase, 'recording')
    assert.equal(recorder.state.get().error, '')
    same(notices(calls), [{ kind: 'warning', message: noBudgetWarning(DESKTOP_TEXT) }])
  }
  {
    const { recorder, capture, ctx, calls } = await setup({ options: { maxPollFailures: 2 } })
    ctx.statuses = [{ ok: false, error: 'cli_failed' }, new Error(IPC_FAILURE)]
    await recorder.start()
    feed(capture)
    await recorder.send({ analyze: true })
    await recorder.polling
    const text = `Статус разбора недоступен: no_answer — ${DESKTOP_TEXT}`
    const job = recorder.job.get()
    same([job.jobId, job.text, job.final, job.stopped], [JOB, text, false, true])
    same(notices(calls).at(-1), { kind: 'warning', message: text })
  }
})

test('the Desktop error text is one line of at most 300 characters, and an empty one names no answer', async () => {
  for (const [failure, expected] of [
    [new Error(`Error invoking remote method 'hermes:api': Error: line one\n\n  line   two`), 'line one line two'],
    [new Error(`Error: ${'x'.repeat(400)}`), 'x'.repeat(300)],
    [new Error(''), 'нет ответа сервера'],
    [undefined, 'нет ответа сервера']
  ]) {
    const { recorder, ctx, calls } = await setup()
    ctx.rest = async () => { throw failure }
    await recorder.start()
    assert.equal(recorder.state.get().phase, 'recording')
    same(notices(calls), [{ kind: 'warning', message: noBudgetWarning(expected) }])
  }
})

test('a window on another connection refuses wrong_connection without a POST, and Send then finishes', async () => {
  const { recorder, calls, capture, ctx, focus } = await setup()
  await recorder.start()
  feed(capture)
  focus.connectionId = 'conn-local'
  await assert.rejects(recorder.send({ analyze: true }), error => error.code === 'wrong_connection')
  assert.match(recorder.state.get().error, /разбор не принят: wrong_connection/)
  assert.equal(named(calls, 'prompt.submit').length, 1)
  assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 0)
  await recorder.send()
  assert.equal(named(calls, 'file.attach').length, 1)
  assert.equal(named(calls, 'prompt.submit').length, 1)
  assert.equal(recorder.state.get().phase, 'idle')
})

test('the job binding names the targetProfile of the route, and a route without one refuses no_target_profile', async () => {
  {
    const { recorder, capture, ctx, remote } = await setup()
    remote.targetProfile = 'remote-default'
    await recorder.start()
    feed(capture)
    await recorder.send({ analyze: true })
    const post = ctx.rests.find(r => r.opts.method === 'POST')
    assert.equal(new URL(`http://plugin${post.path}`).searchParams.get('profile'), 'remote-default')
    assert.equal(recorder.state.get().phase, 'idle')
  }
  {
    const { recorder, capture, ctx, remote, calls } = await setup()
    delete remote.targetProfile
    await recorder.start()
    feed(capture)
    await assert.rejects(recorder.send({ analyze: true }), error => error.code === 'no_target_profile')
    assert.equal(named(calls, 'prompt.submit').length, 1)
    assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 0)
    assert.equal(recorder.state.get().phase, 'ready')
  }
})

test('the window route is checked after the WAV bytes are read, right before POST /jobs', async () => {
  const { recorder, capture, ctx, focus } = await setup()
  await recorder.start()
  feed(capture)
  recorder.stop()
  const blob = recorder.pending.blob
  const read = blob.arrayBuffer.bind(blob)
  blob.arrayBuffer = async () => {
    const bytes = await read()
    focus.connectionId = 'conn-local'
    return bytes
  }
  await assert.rejects(recorder.send({ analyze: true }), error => error.code === 'wrong_connection')
  assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 0)
  assert.match(recorder.state.get().error, /разбор не принят: wrong_connection/)
  focus.connectionId = 'conn-remote'
  await recorder.send({ analyze: true })
  assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 1)
  assert.equal(recorder.state.get().phase, 'idle')
})

test('a retry after the attach finds the route of the attach again and ignores a new focused owner', async () => {
  for (const vanish of [false, true]) {
    const { recorder, host, calls, capture, focus, local, remote } = await setup()
    focus.owner = null
    const request = host.requestProfile
    let refuse = true
    host.requestProfile = (...args) => args[1] === 'prompt.submit' && refuse
      ? Promise.reject(new Error('private-submit-failure'))
      : request(...args)
    await recorder.start()
    feed(capture)
    focus.owner = { connectionId: 'conn-local', profile: 'default' }
    await assert.rejects(recorder.send(), /private-submit-failure/)
    assert.equal(named(calls, 'file.attach')[0][2], local)
    focus.owner = { connectionId: 'conn-remote', profile: 'default' }
    refuse = false
    if (vanish) {
      host.routes = [remote]
      await assert.rejects(recorder.send(), error => error.code === 'ambiguous_route')
      assert.equal(named(calls, 'prompt.submit').length, 0)
      assert.equal(recorder.state.get().phase, 'ready')
      assert.match(recorder.state.get().error, /однозначно определить сервер исходного чата/)
    } else {
      await recorder.send()
      assert.equal(named(calls, 'prompt.submit')[0][2], local)
      assert.equal(named(calls, 'session.resume')[1][2], local)
      assert.equal(recorder.state.get().phase, 'idle')
    }
    assert.equal(named(calls, 'file.attach').length, 1)
  }
})

test('status polling pauses while the window is on another route and resumes on the route of the job', async () => {
  let focusRef
  let naps = 0
  const { recorder, capture, ctx, focus } = await setup({
    options: {
      maxPollFailures: 1,
      sleep: async () => {
        naps++
        if (naps === 1) focusRef.connectionId = 'conn-local'
        if (naps === 3) { focusRef.connectionId = 'conn-remote'; focusRef.profile = 'other' }
        if (naps === 4) focusRef.profile = 'default'
      }
    }
  })
  focusRef = focus
  ctx.statuses = [{ ...status('finished'), result: { job_state: 'finished', status: 'partial' } }]
  await recorder.start()
  feed(capture)
  await recorder.send({ analyze: true })
  await recorder.polling
  assert.equal(naps, 4)
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 1)
  const job = recorder.job.get()
  same([job.jobId, job.text, job.final], [JOB, 'Разбор готов: partial. Итог придёт в чат заметкой', true])
})

test('an intake that is not ready disables Send and analyze with its reason, and Send still works', async () => {
  const { ctx, click, render, calls, capture, recorder } = await setupRegistered()
  ctx.health.intake = { ready: false, error: 'staging_not_configured', detail: 'jobs.staging_root is not set' }
  await click('Записать')
  feed(capture)
  const analyze = findButton(render(), 'Отправить и разобрать')
  assert.equal(analyze.props.disabled, true)
  assert.match(analyze.props.title, /staging_not_configured/)
  await click('Отправить')
  assert.equal(named(calls, 'prompt.submit').length, 1)
  assert.equal(ctx.rests.filter(r => r.opts.method === 'POST').length, 0)
  assert.equal(recorder.state.get().phase, 'idle')
  ctx.disposers.forEach(fn => fn())
})

test('polling sends one request at a time and stops after maxPollFailures failures in a row', async () => {
  const { recorder, capture, ctx, calls } = await setup({ options: { maxPollFailures: 3 } })
  let inFlight = 0
  let most = 0
  const rest = ctx.rest
  ctx.rest = async (path, opts) => {
    inFlight++
    most = Math.max(most, inFlight)
    await tick()
    try { return await rest(path, opts) } finally { inFlight-- }
  }
  ctx.statuses = [new Error('private-offline'), status('queued'), new Error('private-offline'), { ok: false, error: 'cli_failed' }, { ok: false, error: 'cli_timeout' }, status('queued')]
  await recorder.start()
  feed(capture)
  await recorder.send({ analyze: true })
  await recorder.polling
  assert.equal(most, 1)
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 5)
  const job = recorder.job.get()
  same([job.jobId, job.text, job.final, job.stopped], [JOB, 'Статус разбора недоступен: cli_timeout', false, true])
  same(notices(calls).at(-1), { kind: 'warning', message: 'Статус разбора недоступен: cli_timeout' })
})

test('cancel leaves an accepted job polled, and dispose stops the polling', async () => {
  const { recorder, capture, ctx } = await setup({ options: { sleep: () => new Promise(resolve => { wake = resolve }) } })
  let wake
  ctx.statuses = [status('queued'), status('queued')]
  await recorder.start()
  feed(capture)
  await recorder.send({ analyze: true })
  recorder.cancel()
  await tick()
  wake()
  await tick()
  await tick()
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 1)
  recorder.dispose()
  wake()
  await recorder.polling
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 1)
  assert.equal(recorder.job.get().text, 'Разбор: в очереди')
})

test('register adds health, the recorder to composer actions and the job card above the input, and dispose stops it', async () => {
  const timers = []
  const { ctx, click, render, card, capture, calls, recorder } = await setupRegistered({
    globals: { setTimeout: (fn, ms) => timers.push({ fn, ms }), clearTimeout: () => {} }
  })
  same(ctx.many, [])
  same(ctx.registered.map(c => [c.id, c.area]), [['health', 'palette'], ['record', 'composer.actions'], ['job-card', 'composer.top']])
  await click('Записать')
  assert.match(textContent(render()), /00:00 \/ 03:00/)
  assert.ok(findButton(render(), 'Отправить'))
  assert.ok(findButton(render(), 'Отмена'))
  feed(capture)
  recorder.stop()
  assert.match(textContent(render()), /К отправке/)
  await click('Отправить и разобрать')
  assert.equal(named(calls, 'prompt.submit').length, 1)
  assert.doesNotMatch(textContent(render()), /Разбор/)
  assert.match(textContent(card()), /Разбор: принят/)
  await click('Записать')
  ctx.disposers.forEach(fn => fn())
  assert.equal(capture.tracksStopped, 2)
  for (const timer of timers.filter(t => t.ms === 3000)) timer.fn()
  await tick()
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 0)
  same(calls.filter(call => ['retainSession', 'composer.submit'].includes(call[0])), [])
})

test('an SDK without the stock route calls is refused visibly and keeps health', async () => {
  for (const remove of [
    host => { delete host.requestProfile },
    host => { delete host.profileRoutes },
    host => { delete host.state.focusedSessionOwner }
  ]) {
    const { ctx, calls } = await setupRegistered({ host: remove })
    same(ctx.registered.map(c => c.id), ['health'])
    assert.equal(notices(calls).filter(n => n.kind === 'error').length, 1)
  }
})

test('the window warns before it discards a recording and drops the warning on dispose', async () => {
  const listeners = new Map()
  const { ctx, click } = await setupRegistered({
    globals: {
      addEventListener: (name, fn) => listeners.set(name, fn),
      removeEventListener: (name, fn) => { if (listeners.get(name) === fn) listeners.delete(name) }
    }
  })
  const event = { prevented: false, preventDefault() { this.prevented = true } }
  listeners.get('beforeunload')(event)
  assert.equal(event.prevented, false)
  await click('Записать')
  listeners.get('beforeunload')(event)
  assert.equal(event.prevented, true)
  ctx.disposers.forEach(fn => fn())
  assert.equal(listeners.has('beforeunload'), false)
})

test('record errors notify once without raw failures', async () => {
  const { ctx, click, focus, env, capture, calls, host, recorder } = await setupRegistered()
  const expectError = () => {
    same(calls.filter(call => call[0] === 'notify'), [['notify', { kind: 'error', message: recorder.state.get().error }]])
    assert.ok(recorder.state.get().error)
    assert.equal(recorder.state.get().error.includes('private-raw-failure'), false)
    calls.length = 0
  }
  try {
    focus.stored = null
    await click('Записать')
    assert.equal(capture.constraints, null)
    expectError()
    focus.stored = 'stored-a'
    const getUserMedia = env.navigator.mediaDevices.getUserMedia
    env.navigator.mediaDevices.getUserMedia = async () => {
      const error = new Error('private-raw-failure')
      error.name = 'NotAllowedError'
      throw error
    }
    await click('Записать')
    assert.match(recorder.state.get().error, /Нет доступа к микрофону/)
    expectError()
    env.navigator.mediaDevices.getUserMedia = getUserMedia
    await click('Записать')
    feed(capture)
    host.requestProfile = async () => { throw new Error('private-raw-failure') }
    await click('Отправить')
    assert.equal(recorder.state.get().phase, 'ready')
    assert.ok(recorder.pending.blob.size > 44)
    expectError()
    recorder.cancel()
    same(calls.filter(call => call[0] === 'notify'), [])
  } finally {
    ctx.disposers.forEach(fn => fn())
  }
})

test('the plugin source holds no non-stock call, no directive, no Markdown or HTML rendering and no spinner', async () => {
  const source = await readFile(RECORDER_PATH, 'utf8')
  for (const banned of [
    /retainSession/, /composer\.submit/, /\bfetch\(/, /XMLHttpRequest/, /WebSocket/, /\beval\b/, /upload:/,
    /sound-scout-report/, /SPIKE_JOB/, /TRANSCRIPT_DIRECTIVE_AREA/, /MessageTextContent/, /dangerouslySetInnerHTML/,
    /innerHTML/, /GlyphSpinner/, /\bLoader\b/, /@keyframes/
  ]) {
    assert.doesNotMatch(source, banned)
  }
  const imports = [...source.matchAll(/^\s*import\b.*?from\s+['"]([^'"]+)['"]/gm)].map(m => m[1])
  same(imports, ['@hermes/plugin-sdk', 'react/jsx-runtime'])
  const rpc = new Set([...source.matchAll(/'((?:session|file|prompt)\.[a-z_]+)'/g)].map(m => m[1]))
  same([...rpc].sort(), ['file.attach', 'prompt.submit', 'session.interrupt', 'session.resume'])
})

// --- the job card (#108) -------------------------------------------------------------------------

const SECOND = 'b'.repeat(32)
const STAGES = ['prepare', 'asr', 'acoustics', 'audio_review']
const view = (state, extra = {}, jobId = JOB) => ({
  ok: true,
  job: {
    job_id: jobId, state, stages: STAGES, finished_stages: [], current_stage: null, state_reason: null,
    attempts: state === 'queued' ? 0 : 1, cancel_requested: false, staging_removed_at: null, ...extra
  },
  result: null
})
const finishedView = (extra = {}) => ({ ...view('finished', extra), result: { job_id: JOB, job_state: 'finished', attempt: 1, status: 'partial' } })
const reportAnswer = (markdown, extra = {}) => ({
  ok: true, report: { job_id: JOB, job_state: 'finished', attempt: 1, run_id: 'r', run_dir: 'runs/r', markdown, size_bytes: markdown.length, truncated: false, ...extra }
})
const actionRests = ctx => ctx.rests.filter(r => /\/(report|cancel|retry)$/.test(r.path))
const getRests = (ctx, jobId = JOB) => ctx.rests.filter(r => r.path === `/jobs/${jobId}`)

// Each sleep of the poll waits until the test wakes every waiting poll at once.
function gate() {
  let waiting = []
  return {
    sleep: () => new Promise(resolve => waiting.push(resolve)),
    async wake() {
      const now = waiting
      waiting = []
      now.forEach(resolve => resolve())
      for (let i = 0; i < 4; i++) await tick()
    }
  }
}

async function analyzed(context) {
  await context.recorder.start()
  feed(context.capture)
  await context.recorder.send({ analyze: true })
}

test('criterion 1: the card shows the finished, current and waiting stages by polling, and no chat message per stage', async () => {
  const context = await setup()
  const { recorder, ctx, calls, card } = context
  const done = [{ stage: 'prepare', status: 'completed' }]
  ctx.statuses = [
    view('queued', { attempts: 1, finished_stages: [{ stage: 'prepare', status: 'completed' }, { stage: 'asr', status: 'failed' }] }),
    view('running', { attempts: 2, finished_stages: done, current_stage: { stage: 'asr', started_at: 't' } }),
    view('running', { attempts: 2, finished_stages: [...done, { stage: 'asr', status: 'unavailable' }], current_stage: { stage: 'acoustics', started_at: 't' } }),
    view('failed', { attempts: 2, finished_stages: [...done, { stage: 'asr', status: 'skipped' }, { stage: 'acoustics', status: 'failed' }], state_reason: 'analysis_failed' })
  ]
  const seen = []
  recorder.job.subscribe(() => {
    const tree = card()
    if (tree) seen.push({ stages: stageTexts(tree), text: textContent(tree) })
  })
  await analyzed(context)
  await recorder.polling
  same(seen.map(item => item.stages), [
    [],
    ['Подготовка: ждёт', 'Распознавание речи: ждёт', 'Измерения: ждёт', 'Слуховой анализ: ждёт'],
    ['Подготовка: готово', 'Распознавание речи: идёт', 'Измерения: ждёт', 'Слуховой анализ: ждёт'],
    ['Подготовка: готово', 'Распознавание речи: недоступен', 'Измерения: идёт', 'Слуховой анализ: ждёт'],
    ['Подготовка: готово', 'Распознавание речи: пропущен', 'Измерения: ошибка', 'Слуховой анализ: не выполнен']
  ])
  assert.match(seen[1].text, /Разбор: в очереди/)
  assert.match(seen[1].text, /Попытка 2/)
  assert.match(seen[4].text, /Разбор не удался: analysis_failed/)
  same(calls.filter(call => ['session.resume', 'file.attach', 'prompt.submit', 'retain'].includes(call[0])).map(call => call[0]),
    ['retain', 'session.resume', 'file.attach', 'prompt.submit'])
  assert.equal(getRests(ctx).length, 4)
  assert.equal(nodes(nodes(card(), 'ol')[0], 'Codicon').length, 4)
  assert.ok(nodes(card(), 'Codicon').every(icon => icon.props['aria-hidden'] === true))
})

test('the stage rows follow the plan order and name an unknown stage by its name', async () => {
  const plugin = await loadPlugin({ path: RECORDER_PATH })
  same(plugin.stageRows(null), [])
  same(plugin.stageRows({ state: 'cancelled', stages: ['prepare', 'decode'], finished_stages: [{ stage: 'decode', status: 'completed' }] }), [
    { stage: 'prepare', label: 'Подготовка', mark: 'missing', status: 'missing' },
    { stage: 'decode', label: 'decode', mark: 'done', status: 'completed' }
  ])
})

test('the card renders only in the composer of the chat of the job', async () => {
  const context = await setup()
  const { recorder, ctx, composer, card, focus } = context
  ctx.statuses = [finishedView()]
  assert.equal(card(), null)
  await analyzed(context)
  focus.stored = 'stored-b'
  await recorder.polling
  assert.equal(recorder.job.get().sessionId, 'stored-a')
  assert.match(textContent(card()), /Разбор готов: partial/)
  composer.stored = 'stored-b'
  assert.equal(card(), null)
  composer.stored = null
  assert.equal(card(), null)
})

test('criterion 2: a click opens the full report from the server as a plain string in an SDK Dialog', async () => {
  const context = await setup()
  const { recorder, ctx, card, press } = context
  ctx.statuses = [finishedView()]
  await analyzed(context)
  await recorder.polling
  const markdown = '# Отчёт\n::foo{x="1"} <b>жирный</b> MEDIA:/etc/passwd\n'
  let finish
  ctx.actions = [() => new Promise(resolve => { finish = resolve })]
  const opening = press('Открыть отчёт')
  await tick()
  const loading = card()
  assert.match(textContent(nodes(loading, 'Dialog')[0]), /Загрузка отчёта…/)
  assert.ok(nodes(loading, 'Button').every(button => button.props.disabled === true))
  finish(reportAnswer(markdown))
  await opening
  same(actionRests(ctx).map(r => [r.path, r.opts]), [[`/jobs/${JOB}/report`, { timeoutMs: 10000 }]])
  const tree = card()
  const [dialog] = nodes(tree, 'Dialog')
  assert.equal(dialog.props.open, true)
  assert.equal(textContent(nodes(tree, 'DialogTitle')[0]), 'Отчёт Sound Scout')
  assert.equal(textContent(nodes(tree, 'DialogDescription')[0]), 'Попытка 1 · partial')
  const [pre] = nodes(tree, 'pre')
  assert.equal(pre.props.children, markdown)
  assert.equal(typeof pre.props.children, 'string')
  assert.equal(pre.props.tabIndex, 0)
  assert.equal(pre.props['aria-label'], 'Текст отчёта')
  assert.equal(pre.props.className, 'sound-scout-card__report')
  assert.equal(nodes(tree, 'b').length, 0)
  assert.doesNotMatch(textContent(tree), /256 KiB/)
  dialog.props.onOpenChange(false)
  assert.equal(nodes(card(), 'Dialog').length, 0)

  ctx.actions = [reportAnswer('x\n', { truncated: true, size_bytes: 300 * 1024 })]
  await press('Открыть отчёт')
  assert.match(textContent(card()), /Показаны первые 256 KiB\. Полный отчёт: `runs\/r\/report\.md` в data root на станции/)
  assert.doesNotMatch(textContent(card()), /jobs report/)
  recorder.closeReport()

  ctx.actions = [{ ok: false, error: 'no_result', detail: 'private-detail' }]
  await press('Открыть отчёт')
  const [alert] = [...walk(nodes(card(), 'Dialog')[0])].filter(node => node.props?.role === 'alert')
  assert.equal(textContent(alert), 'Отчёт недоступен: у разбора пока нет результата.')
  assert.doesNotMatch(textContent(card()), /private-/)
  assert.equal(actionRests(ctx).length, 3, 'no cache: every click asks the server')
})

test('criterion 3: cancel acts on the job of the card only and polls that job at once', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, card, press } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await poll.wake()
  ctx.post = { ok: true, job_id: SECOND, reused: false }
  await analyzed(context)
  ctx.statuses = [view('running', {}, SECOND)]
  await poll.wake()
  assert.equal(recorder.job.get().jobId, SECOND)
  assert.ok(findButton(card(), 'Отменить разбор'))
  const requested = view('running', { cancel_requested: true }, SECOND)
  ctx.actions = [{ ok: true, job: { ...requested.job, cancel: 'requested' } }]
  ctx.statuses = [requested]
  const before = ctx.rests.length
  await press('Отменить разбор')
  await tick()
  same(ctx.rests.slice(before).map(r => [r.opts.method ?? 'GET', r.path]), [['POST', `/jobs/${SECOND}/cancel`], ['GET', `/jobs/${SECOND}`]])
  same(ctx.rests.slice(before)[0].opts, { method: 'POST', body: {}, timeoutMs: 10000 })
  assert.equal(ctx.rests.filter(r => r.path.startsWith(`/jobs/${JOB}/`)).length, 0)
  assert.match(textContent(card()), /отмена запрошена/)
  assert.equal(findButton(card(), 'Отменить разбор'), undefined)

  ctx.actions = [{ ok: true, job: { ...view('cancelled', {}, SECOND).job, cancel: 'cancelled' } }]
  ctx.statuses = [view('cancelled', {}, SECOND)]
  recorder.setJob({ view: view('queued', {}, SECOND).job })
  await press('Отменить разбор')
  await recorder.polling
  assert.match(textContent(card()), /Разбор отменён/)
  assert.ok(findButton(card(), 'Повторить разбор'))
  assert.ok(findButton(card(), 'Скрыть'))
})

// #188: «Отменить разбор» also stops the running Hermes turn of the chat of the job.
const cancelled = (jobId = JOB) => ({ ok: true, job: { ...view('cancelled', {}, jobId).job, cancel: 'cancelled' } })
const stopCalls = (calls, from) => calls.slice(from).filter(call => ['retain', 'release', 'session.resume', 'session.interrupt'].includes(call[0]))
const INTERRUPT_FAILURE = "Error invoking remote method 'hermes:api': Error: Session execution identity unavailable; reconnect before controlling this turn"

test('cancel stops the running turn of the chat of the job with session.interrupt and no turn of another chat', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, calls, host, focus, remote, card } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await poll.wake()
  focus.stored = 'stored-b'
  ctx.post = { ok: true, job_id: SECOND, reused: false }
  ctx.statuses = [view('running', {}, SECOND)]
  await analyzed(context)
  await poll.wake()
  assert.equal(recorder.job.get().sessionId, 'stored-b')
  host.turns = { 'stored-a': { running: true, execution_generation: 5 }, 'stored-b': { running: true, execution_generation: 7 } }
  focus.stored = 'stored-a'
  ctx.actions = [cancelled(SECOND)]
  ctx.statuses = [view('cancelled', {}, SECOND)]
  const before = calls.length
  await recorder.cancelJob()
  await recorder.polling
  same(stopCalls(calls, before).map(call => [call[0], call[1]]), [
    ['retain', remote],
    ['session.resume', { session_id: 'stored-b', omit_messages: true }],
    ['session.interrupt', { session_id: 'runtime-b', execution_generation: 7 }],
    ['release', null]
  ])
  const interrupt = named(calls, 'session.interrupt')
  assert.equal(interrupt.length, 1)
  same(interrupt[0].slice(2), [remote, null, FOREGROUND])
  assert.ok(context.order.indexOf(`rest POST /jobs/${SECOND}/cancel`) < context.order.indexOf('session.interrupt'))
  assert.equal(recorder.job.get().error, '')
  context.composer.stored = 'stored-b'
  assert.match(textContent(card()), /Разбор отменён/)
  assert.equal(nodes(card(), 'p').length, 0)
})

test('cancel stops the turn of the chat of the job when a newer job of another chat took the card meanwhile', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, calls, host, focus, remote } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await poll.wake()
  host.turns = { 'stored-a': { running: true, execution_generation: 5 }, 'stored-b': { running: true, execution_generation: 7 } }
  let finish
  ctx.actions = [() => new Promise(resolve => { finish = resolve })]
  const cancelling = recorder.cancelJob()
  await tick()
  focus.stored = 'stored-b'
  ctx.post = { ok: true, job_id: SECOND, reused: false }
  ctx.statuses = [view('running', {}, SECOND)]
  await analyzed(context)
  await poll.wake()
  assert.equal(recorder.job.get().jobId, SECOND)
  const before = calls.length
  finish(cancelled())
  await cancelling
  same(stopCalls(calls, before).map(call => [call[0], call[1]]), [
    ['retain', remote],
    ['session.resume', { session_id: 'stored-a', omit_messages: true }],
    ['session.interrupt', { session_id: 'runtime-a', execution_generation: 5 }],
    ['release', null]
  ])
  const job = recorder.job.get()
  assert.equal(job.jobId, SECOND)
  assert.equal(job.error, '')
  assert.equal(job.busy, null)
})

test('cancel never stops a turn of its chat when a newer job of the same chat took the card meanwhile', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, calls, host } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await poll.wake()
  host.turns = { 'stored-a': { running: true, execution_generation: 6 } }
  let finish
  ctx.actions = [() => new Promise(resolve => { finish = resolve })]
  const cancelling = recorder.cancelJob()
  await tick()
  ctx.post = { ok: true, job_id: SECOND, reused: false }
  ctx.statuses = [view('running', {}, SECOND)]
  await analyzed(context)
  await poll.wake()
  const job = recorder.job.get()
  assert.equal(job.jobId, SECOND)
  assert.equal(job.sessionId, 'stored-a')
  const before = calls.length
  finish(cancelled())
  await cancelling
  same(stopCalls(calls, before), [])
  assert.equal(named(calls, 'session.interrupt').length, 0)
  assert.equal(recorder.job.get().error, '')
})

test('cancel of a job whose chat has no running turn only cancels the job and shows no error', async () => {
  const context = await setup()
  const { recorder, ctx, calls, host, card } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await recorder.polling
  host.turns = { 'stored-a': { running: false, execution_generation: 2 } }
  ctx.actions = [cancelled()]
  ctx.statuses = [view('cancelled')]
  const before = calls.length
  await recorder.cancelJob()
  await recorder.polling
  same(stopCalls(calls, before).map(call => call[0]), ['retain', 'session.resume', 'release'])
  assert.equal(named(calls, 'session.interrupt').length, 0)
  assert.equal(recorder.job.get().error, '')
  assert.match(textContent(card()), /Разбор отменён/)
  assert.equal(nodes(card(), 'p').length, 0)
  assert.ok(notices(calls).every(notice => notice.kind !== 'error'))
})

test('a failed turn stop keeps the job cancelled and shows the Desktop error text on the card', async () => {
  for (const [name, fail] of [
    ['interrupt', host => { host.turns = { 'stored-a': { running: true, execution_generation: 4 } }; host.interruptFailure = new Error(INTERRUPT_FAILURE) }],
    ['resume', host => {
      const request = host.requestProfile
      host.requestProfile = (...args) => args[1] === 'session.resume' ? Promise.reject(new Error(INTERRUPT_FAILURE)) : request(...args)
    }],
    ['retain', host => { host.retainProfile = async () => { throw new Error(INTERRUPT_FAILURE) } }]
  ]) {
    const context = await setup()
    const { recorder, ctx, calls, host, card } = context
    ctx.statuses = [view('running')]
    await analyzed(context)
    await recorder.polling
    fail(host)
    ctx.actions = [cancelled()]
    ctx.statuses = [view('cancelled')]
    const before = calls.length
    await recorder.cancelJob()
    await recorder.polling
    const job = recorder.job.get()
    assert.equal(job.view.state, 'cancelled', name)
    assert.equal(job.busy, null, name)
    const text = 'Разбор отменён, но ход чата не остановлен: Session execution identity unavailable; reconnect before controlling this turn'
    assert.equal(job.error, text, name)
    const tree = card()
    assert.match(textContent(tree), /Разбор отменён/, name)
    same(nodes(tree, 'p').map(node => [node.props.role, node.props.children]), [['alert', text]])
    assert.ok(findButton(tree, 'Повторить разбор'), name)
    if (name !== 'retain') assert.equal(calls.slice(before).filter(call => call[0] === 'release').length, 1, name)
  }
})

test('a refused job cancel shows its error and stops no turn', async () => {
  const context = await setup()
  const { recorder, ctx, calls, host } = context
  ctx.statuses = [view('running')]
  await analyzed(context)
  await recorder.polling
  host.turns = { 'stored-a': { running: true, execution_generation: 1 } }
  ctx.actions = [{ ok: false, error: 'not_cancellable' }]
  const before = calls.length
  await recorder.cancelJob()
  assert.equal(stopCalls(calls, before).length, 0)
  assert.equal(recorder.job.get().error, 'Отменить разбор не удалось: not_cancellable')
})

test('criterion 3: retry keeps the job_id, shows the attempt and reads the report of the new attempt', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, card, press, calls } = context
  ctx.statuses = [view('failed', { state_reason: 'analysis_failed' })]
  await analyzed(context)
  await poll.wake()
  await recorder.polling
  assert.equal(recorder.job.get().final, true)
  ctx.actions = [{ ok: true, job: { ...view('queued', { attempts: 1 }).job, retry: 'queued', next_attempt: `${JOB}-a2` } }]
  ctx.statuses = [view('queued', { attempts: 1 })]
  await press('Повторить разбор')
  await tick()
  same(actionRests(ctx).map(r => [r.opts.method, r.path]), [['POST', `/jobs/${JOB}/retry`]])
  let text = textContent(card())
  assert.match(text, /Разбор: в очереди/)
  assert.match(text, /Попытка 2/)
  assert.equal(recorder.job.get().final, false)
  ctx.statuses = [view('running', { attempts: 2, current_stage: { stage: 'prepare', started_at: 't' } })]
  await poll.wake()
  text = textContent(card())
  assert.match(text, /Попытка 2/)
  assert.match(text, /Подготовка: идёт/)
  ctx.statuses = [{ ...finishedView({ attempts: 2 }), result: { job_id: JOB, job_state: 'finished', attempt: 2, status: 'complete' } }]
  await poll.wake()
  await recorder.polling
  assert.match(textContent(card()), /Разбор готов: complete/)
  assert.ok(getRests(ctx).every(r => r.path === `/jobs/${JOB}`))
  ctx.actions = [reportAnswer('попытка 2\n', { attempt: 2 })]
  await press('Открыть отчёт')
  assert.equal(nodes(card(), 'pre')[0].props.children, 'попытка 2\n')
  assert.equal(textContent(nodes(card(), 'DialogDescription')[0]), 'Попытка 2 · complete')
  same(actionRests(ctx).map(r => r.path), [`/jobs/${JOB}/retry`, `/jobs/${JOB}/report`])
  assert.equal(named(calls, 'prompt.submit').length, 1)
})

test('a refused retry names why in Russian after one status request, and a removed input hides the retry button', async () => {
  const removed = { staging_removed_at: '2026-11-23T10:00:00Z' }
  for (const [error, after, expected, buttons] of [
    ['not_retryable', view('failed', { state_reason: 'x', ...removed }), 'Повтор невозможен: исходная запись уже удалена. Запиши заново.', ['Скрыть']],
    ['not_retryable', view('failed', { state_reason: 'x' }), 'Повторить разбор не удалось: not_retryable', ['Повторить разбор', 'Скрыть']],
    ['queue_full', view('failed', { state_reason: 'x' }), 'Очередь занята, повтори позже.', ['Повторить разбор', 'Скрыть']],
    ['staging_full', view('failed', { state_reason: 'x' }), 'Очередь занята, повтори позже.', ['Повторить разбор', 'Скрыть']],
    ['cli_timeout', view('failed', { state_reason: 'x' }), 'Повторить разбор не удалось: cli_timeout', ['Повторить разбор', 'Скрыть']]
  ]) {
    const context = await setup()
    context.ctx.statuses = [view('failed', { state_reason: 'x' })]
    await analyzed(context)
    await context.recorder.polling
    context.ctx.actions = [{ ok: false, error, detail: 'private-detail' }]
    context.ctx.statuses = [after]
    await context.press('Повторить разбор')
    const alerts = [...walk(context.card())].filter(node => node.props?.role === 'alert')
    same(alerts.map(textContent), [expected])
    same(buttonLabels(context.card()), buttons)
    assert.equal(getRests(context.ctx).length, 2)
    assert.equal(context.recorder.job.get().busy, null)
  }
  const context = await setup()
  context.ctx.statuses = [view('cancelled', { staging_removed_at: '2026-11-23T10:00:00Z' })]
  await analyzed(context)
  await context.recorder.polling
  same(buttonLabels(context.card()), ['Скрыть'])
})

test('a failed retry follows the state the job has now, or shows Refresh when the status request fails too', async () => {
  const poll = gate()
  const context = await setup({ options: { sleep: poll.sleep } })
  const { recorder, ctx, card, press } = context
  ctx.statuses = [view('failed', { state_reason: 'x' })]
  await analyzed(context)
  await poll.wake()
  await recorder.polling
  ctx.actions = [new Error('socket hang up')]
  ctx.statuses = [view('queued', { attempts: 1 })]
  await press('Повторить разбор')
  let text = textContent(card())
  assert.match(text, /Разбор: в очереди/)
  assert.match(text, /Повторить разбор не удалось: no_answer — socket hang up/)
  assert.equal(recorder.job.get().final, false)
  ctx.statuses = [view('failed', { attempts: 2, state_reason: 'y' })]
  await poll.wake()
  await recorder.polling
  assert.match(textContent(card()), /Разбор не удался: y/)

  ctx.actions = [{ ok: false, error: 'cli_timeout' }]
  ctx.statuses = [new Error('offline')]
  const before = getRests(ctx).length
  await press('Повторить разбор')
  assert.equal(getRests(ctx).length, before + 1)
  assert.equal(recorder.job.get().stopped, true)
  same(buttonLabels(card()), ['Повторить разбор', 'Обновить', 'Скрыть'])
  ctx.statuses = [view('failed', { attempts: 2, state_reason: 'y' })]
  await press('Обновить')
  await recorder.polling
  assert.equal(recorder.job.get().stopped, false)
})

test('a second click while an action is in flight sends one request, and another connection sends none', async () => {
  const context = await setup()
  const { recorder, ctx, card, press, focus } = context
  ctx.statuses = [view('failed', { state_reason: 'x' })]
  await analyzed(context)
  await recorder.polling
  let finish
  ctx.actions = [() => new Promise(resolve => { finish = resolve })]
  const first = press('Повторить разбор')
  await tick()
  assert.equal(recorder.job.get().busy, 'retry')
  assert.ok(nodes(card(), 'Button').every(button => button.props.disabled === true))
  await Promise.all([recorder.retryJob(), recorder.cancelJob(), recorder.openReport()])
  finish({ ok: false, error: 'queue_full' })
  await first
  assert.equal(actionRests(ctx).length, 1)
  focus.connectionId = 'conn-local'
  const before = ctx.rests.length
  await press('Повторить разбор')
  assert.equal(ctx.rests.length, before)
  const alerts = [...walk(card())].filter(node => node.props?.role === 'alert')
  same(alerts.map(textContent), ['Окно подключено к другому серверу. Переключись на подключение этого чата.'])
})

test('a poll answer about another job is bad_answer and changes nothing', async () => {
  const context = await setup({ options: { maxPollFailures: 1 } })
  const { recorder, ctx, card } = context
  ctx.statuses = [finishedView({}, SECOND), finishedView()]
  ctx.statuses[0].job.job_id = SECOND
  await analyzed(context)
  await recorder.polling
  const job = recorder.job.get()
  same([job.view, job.result, job.final, job.stopped, job.text], [null, null, false, true, 'Статус разбора недоступен: bad_answer'])
  same(buttonLabels(card()), ['Обновить', 'Скрыть'])
  await context.press('Обновить')
  await recorder.polling
  assert.match(textContent(card()), /Разбор готов: partial/)
  await context.press('Скрыть')
  assert.equal(recorder.job.get(), null)
  assert.equal(card(), null)
})

const COLOR_PROPERTY = /^(color|background(-color)?|border(-(top|right|bottom|left))?(-color)?|outline(-color)?|fill|stroke)$/
const REDUCED_MOTION = /@media \(prefers-reduced-motion: reduce\) \{[^{}]*\{ transition: none !important; animation: none !important; \} \}/g

function assertCriterion4(tree, where) {
  for (const node of walk(tree)) {
    assert.ok(!['button', 'dialog', 'input', 'a', 'img', 'svg'].includes(node.type), `${where}: raw <${node.type}>`)
    assert.equal(node.props?.dangerouslySetInnerHTML, undefined, where)
    for (const key of Object.keys(node.props?.style ?? {})) assert.doesNotMatch(key, /transition|animation/i, where)
    if (node.type === 'Button') {
      assert.equal(node.props.type, 'button', where)
      assert.ok(typeof node.props['aria-label'] === 'string' && node.props['aria-label'], where)
      assert.ok(node.props.title, where)
    }
  }
  return nodes(tree, 'style').map(node => node.props.children)
}

test('criterion 4: SDK components, var(--ui-*) colors, typed and labelled buttons and reduced motion', async () => {
  const styles = new Set()
  {
    const context = await setup()
    const { recorder, ctx, card } = context
    ctx.statuses = [finishedView()]
    await analyzed(context)
    await recorder.polling
    const scenes = [
      ['queued', { view: view('queued').job, final: false }],
      ['running', { view: view('running', { finished_stages: [{ stage: 'prepare', status: 'completed' }], current_stage: { stage: 'asr', started_at: 't' } }).job, final: false }],
      ['failed', { view: view('failed', { state_reason: 'x' }).job, final: true, error: 'Очередь занята, повтори позже.' }],
      ['stopped', { stopped: true }],
      ['finished', { view: finishedView().job, result: finishedView().result, final: true, stopped: false, error: '' }],
      ['loading', { report: { open: true, loading: true, markdown: '', truncated: false, attempt: null, error: '' } }],
      ['report', { report: { open: true, loading: false, markdown: 'text', truncated: true, attempt: 1, error: '' } }],
      ['report error', { report: { open: true, loading: false, markdown: '', truncated: false, attempt: null, error: 'x' } }]
    ]
    for (const [name, values] of scenes) {
      recorder.setJob(values)
      const tree = card()
      assert.equal(tree.type, 'section', name)
      assert.equal(tree.props['aria-label'], 'Разбор Sound Scout', name)
      const [status] = [...walk(tree)].filter(node => node.props?.role === 'status')
      assert.equal(status.props['aria-live'], 'polite', name)
      assertCriterion4(tree, name).forEach(css => styles.add(css))
    }
  }
  {
    const { recorder, render, ctx, capture } = await setupRegistered({
      globals: { setTimeout: () => 1, clearTimeout: () => {} }
    })
    for (const phase of ['idle', 'starting', 'recording', 'ready', 'sending']) {
      if (phase === 'recording') {
        recorder.state.set({ ...recorder.state.get(), phase: 'idle' })
        await recorder.start()
        feed(capture)
      } else {
        recorder.state.set({ ...recorder.state.get(), phase, error: phase === 'ready' ? 'Запись сохранена.' : '' })
      }
      assertCriterion4(render(), phase).forEach(css => styles.add(css))
    }
    recorder.cancel()
    ctx.disposers.forEach(fn => fn())
  }
  assert.equal(styles.size, 2, 'the recorder sheet and the card sheet')
  for (const css of styles) {
    assert.equal(css.match(REDUCED_MOTION)?.length, 1, css.slice(0, 60))
    assert.doesNotMatch(css, /#[0-9a-f]{3,8}\b|rgba?\(|hsla?\(|currentColor/i)
    let checked = 0
    for (const [, property, value] of css.matchAll(/([a-z-]+)\s*:\s*([^;{}]+)/g)) {
      if (!COLOR_PROPERTY.test(property)) continue
      checked++
      assert.match(value, /var\(--ui-/, `${property}: ${value}`)
    }
    assert.ok(checked > 3)
    assert.doesNotMatch(css.replace(REDUCED_MOTION, ''), /transition|animation|@keyframes/)
  }
  const card = [...styles].find(css => css.includes('.sound-scout-card {'))
  for (const token of ['--ui-bg-elevated', '--ui-stroke-secondary', '--ui-text-primary', '--ui-text-secondary', '--ui-green', '--ui-yellow', '--ui-red']) {
    assert.ok(card.includes(`var(${token})`), token)
  }
  assert.match(card, /\.sound-scout-card__report \{[^}]*white-space:pre-wrap;[^}]*\}/)
  assert.match(card, /\.sound-scout-card__report \{[^}]*max-height:70vh;[^}]*overflow:auto;/)
})

test('register leaves the transcript directive out, and an SDK without useComposerSession keeps the recorder without the card', async () => {
  {
    const { ctx } = await setupRegistered()
    assert.equal(ctx.registered.some(item => item.area === 'transcript.directive'), false)
    ctx.disposers.forEach(fn => fn())
  }
  {
    const { ctx, calls } = await setupRegistered({ sdk: { useComposerSession: undefined } })
    same(ctx.registered.map(c => [c.id, c.area]), [['health', 'palette'], ['record', 'composer.actions']])
    same(notices(calls), [])
    ctx.disposers.forEach(fn => fn())
  }
})
