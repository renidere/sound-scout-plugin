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
  if (node?.type === 'button' && textContent(node) === label) return node
  return [].concat(node?.props?.children || []).map(child => findButton(child, label)).find(Boolean)
}

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
      if (method === 'session.resume') return { session_id: 'runtime-a', stored_session_id: params.session_id }
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
    async rest(path, opts = {}) {
      ctx.rests.push({ path, opts })
      order.push(`rest ${opts.method || 'GET'} ${path.split('?')[0]}`)
      const answer = path === '/health' ? ctx.health
        : path.startsWith('/jobs?') ? ctx.post
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

async function setup({ globals = {}, options = {} } = {}) {
  const order = []
  const { env, capture } = fakeAudio()
  const plugin = await loadPlugin({ path: RECORDER_PATH, globals: { ...env, ...globals } })
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
  return { plugin, ...fake, ctx, recorder, env, capture, sleeps, order }
}

async function setupRegistered({ globals = {}, host: patch } = {}) {
  const order = []
  const { env, capture } = fakeAudio()
  const fake = fakeHost(order)
  patch?.(fake.host)
  const plugin = await loadPlugin({ path: RECORDER_PATH, sdk: { host: fake.host }, globals: { ...env, ...globals } })
  const ctx = fakeCtx(order)
  plugin.default.register(ctx)
  const element = ctx.registered[0]?.render()
  const render = () => element.type(element.props)
  const click = label => findButton(render(), label).props.onClick()
  return { plugin, ...fake, ctx, env, capture, element, render, click, recorder: element?.props.recorder, order }
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

test('no budget from GET /health means no recording', async () => {
  const answers = [
    { ok: false, error: 'cli_failed', detail: 'x' },
    { ok: true, budgets: null, intake: { ready: false, error: 'cli_failed', detail: 'private-detail' } },
    { ok: true, budgets: { max_duration_seconds: 0, max_input_bytes: 10 }, intake: { ready: true } },
    { ok: true, budgets: { max_duration_seconds: 1.5, max_input_bytes: 10 }, intake: { ready: true } }
  ]
  for (const answer of answers) {
    const { recorder, capture, ctx, calls } = await setup()
    ctx.health = answer
    await recorder.start()
    assert.equal(capture.constraints, null)
    assert.equal(recorder.state.get().phase, 'idle')
    assert.match(recorder.state.get().error, /Sound Scout не сообщил лимит записи/)
    assert.doesNotMatch(recorder.state.get().error, /private-/)
    assert.equal(notices(calls).length, 1)
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
  same(recorder.job.get(), { jobId: JOB, text: 'Разбор не удался: input_invalid', final: true })
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
    const { recorder, capture, ctx } = await setup()
    ctx.health = new Error(IPC_FAILURE)
    await recorder.start()
    assert.equal(capture.constraints, null)
    assert.equal(recorder.state.get().error, `Sound Scout не сообщил лимит записи: ${DESKTOP_TEXT}.`)
  }
  {
    const { recorder, capture, ctx, calls } = await setup({ options: { maxPollFailures: 2 } })
    ctx.statuses = [{ ok: false, error: 'cli_failed' }, new Error(IPC_FAILURE)]
    await recorder.start()
    feed(capture)
    await recorder.send({ analyze: true })
    await recorder.polling
    const text = `Статус разбора недоступен: no_answer — ${DESKTOP_TEXT}`
    same(recorder.job.get(), { jobId: JOB, text, final: true })
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
    const { recorder, ctx } = await setup()
    ctx.rest = async () => { throw failure }
    await recorder.start()
    assert.equal(recorder.state.get().error, `Sound Scout не сообщил лимит записи: ${expected}.`)
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
  same(recorder.job.get(), { jobId: JOB, text: 'Разбор готов: partial. Итог придёт в чат заметкой', final: true })
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
  same(recorder.job.get(), { jobId: JOB, text: 'Статус разбора недоступен: cli_timeout', final: true })
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

test('register keeps health and the directive, adds the recorder to composer actions and dispose stops it', async () => {
  const timers = []
  const { ctx, click, render, capture, calls, recorder } = await setupRegistered({
    globals: { setTimeout: (fn, ms) => timers.push({ fn, ms }), clearTimeout: () => {} }
  })
  same(ctx.many.map(c => [c.id, c.area]), [['health', 'palette'], ['report-directive', 'transcript.directive']])
  same(ctx.registered.map(c => [c.id, c.area]), [['record', 'composer.actions']])
  await click('Записать')
  assert.match(textContent(render()), /00:00 \/ 03:00/)
  assert.ok(findButton(render(), 'Отправить'))
  assert.ok(findButton(render(), 'Отмена'))
  feed(capture)
  recorder.stop()
  assert.match(textContent(render()), /К отправке/)
  await click('Отправить и разобрать')
  assert.equal(named(calls, 'prompt.submit').length, 1)
  assert.match(textContent(render()), /Разбор: принят/)
  await click('Записать')
  ctx.disposers.forEach(fn => fn())
  assert.equal(capture.tracksStopped, 2)
  for (const timer of timers.filter(t => t.ms === 3000)) timer.fn()
  await tick()
  assert.equal(ctx.rests.filter(r => r.path === `/jobs/${JOB}`).length, 0)
  same(calls.filter(call => ['retainSession', 'composer.submit'].includes(call[0])), [])
})

test('an SDK without the stock route calls is refused visibly and keeps health and the directive', async () => {
  for (const remove of [
    host => { delete host.requestProfile },
    host => { delete host.profileRoutes },
    host => { delete host.state.focusedSessionOwner }
  ]) {
    const { ctx, calls } = await setupRegistered({ host: remove })
    same(ctx.registered, [])
    assert.equal(ctx.many.length, 2)
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

test('the plugin source holds no non-stock call and only the three backend routes', async () => {
  const source = await readFile(RECORDER_PATH, 'utf8')
  for (const banned of [/retainSession/, /composer\.submit/, /\bfetch\(/, /XMLHttpRequest/, /WebSocket/, /\beval\b/, /upload:/]) {
    assert.doesNotMatch(source, banned)
  }
  const imports = [...source.matchAll(/^\s*import\b.*?from\s+['"]([^'"]+)['"]/gm)].map(m => m[1])
  same(imports, ['@hermes/plugin-sdk', 'react/jsx-runtime'])
  const rpc = new Set([...source.matchAll(/'((?:session|file|prompt)\.[a-z_]+)'/g)].map(m => m[1]))
  same([...rpc].sort(), ['file.attach', 'prompt.submit', 'session.resume'])
})
