// Behaviour of the Spike A probe against a fake SDK. It proves the probe's contract only, not what
// the real Hermes Desktop or gateway do: that is the owner's run (decision 0016).
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { loadPlugin, PROBE_PATH } from './load_plugin.mjs'

const RAW_CONN = 'station-registry-7f3a9c'
const ROUTE = { connectionId: RAW_CONN, mode: 'remote', profile: 'default', targetProfile: 'default' }
const OTHER_ROUTE = { connectionId: 'laptop-local', mode: 'local', primary: true, profile: 'default', targetProfile: 'default' }
const PRIVATE_TEXT = 'private earlier message'
const WAV_BYTES = 5760044

const LABELS = {
  'snapshot-old': 'Probe A: snapshot — old chat',
  'snapshot-renamed': 'Probe A: snapshot — renamed chat',
  'snapshot-new': 'Probe A: snapshot — new chat',
  'mark-target': 'Probe A: mark this chat as test target',
  attach: 'Probe A: attach synthetic 30 s WAV to target',
  'prompt-submit': 'Probe A: prompt.submit to target',
  'composer-submit': 'Probe A: composer.submit to target',
  'copy-log': 'Probe A: copy log'
}
const WRITE_COMMANDS = ['attach', 'prompt-submit', 'composer-submit']

function atom(value) {
  const listeners = new Set()
  return {
    get: () => value,
    set(next) {
      value = next
      for (const fn of [...listeners]) fn(next)
    },
    listen(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
    listeners
  }
}

function makeHost({
  routes = [ROUTE],
  owner = { connectionId: RAW_CONN, profile: 'default' },
  submitCopies = 1,
  replayNewAdmission = false,
  composerAck = true,
  composerCopies = 1,
  fail = {},
  failVerify = null,
  verifyReply = null,
  failAfterWrite = null,
  attachReply = null,
  retainGate = null
} = {}) {
  const calls = []
  const history = [{ role: 'user', content: PRIVATE_TEXT }, { role: 'assistant', content: [{ type: 'text', text: 'private answer' }] }]
  const admissions = new Map()
  const host = {
    calls,
    history,
    released: 0,
    notes: [],
    state: {
      focusedStoredSessionId: atom('stored-1'),
      focusedSessionOwner: atom(owner),
      focusedSessionProfile: atom('default'),
      focusedSessionId: atom('runtime-1'),
      connectionId: atom(RAW_CONN),
      gateway: atom('open'),
      profile: atom('default')
    },
    async profileRoutes() {
      calls.push({ fn: 'profileRoutes' })
      return routes
    },
    async retainProfile(route, options) {
      calls.push({ fn: 'retainProfile', route, options })
      if (retainGate) await retainGate
      return () => { host.released++ }
    },
    async requestProfile(...args) {
      const [route, method, params, timeout, options] = args
      calls.push({ fn: 'requestProfile', route, method, params, timeout, options, argc: args.length })
      if (fail[method] && (method !== 'session.resume' || params.omit_messages)) throw new Error(fail[method])
      if (method === 'session.resume' && !params.omit_messages) {
        if (failVerify) throw new Error(failVerify)
        if (verifyReply) return verifyReply
      }
      if (method === 'session.resume') {
        return params.omit_messages
          ? { session_id: 'runtime-1', messages: [], messages_omitted: true, message_count: history.length }
          : { session_id: 'runtime-1', messages: structuredClone(history), message_count: history.length }
      }
      if (method === 'file.attach') {
        return attachReply ?? {
          attached: true,
          name: params.name,
          path: `/home/someone/hermes-home/attachments/${params.name}`,
          ref_path: `/home/someone/hermes-home/attachments/${params.name}`,
          ref_text: `@file:/home/someone/hermes-home/attachments/${params.name}`,
          uploaded: true
        }
      }
      if (method === 'prompt.submit') {
        if (!admissions.has(params.submission_id)) {
          admissions.set(params.submission_id, `adm-${admissions.size + 1}`)
          for (let i = 0; i < submitCopies; i++) history.push({ role: 'user', content: params.text })
        }
        if (failAfterWrite) throw new Error(failAfterWrite)
        const id = replayNewAdmission ? `adm-fresh-${calls.length}` : admissions.get(params.submission_id)
        return { admission_id: id, ref: 'r', sequence: 1, status: 'queued', outcome: null }
      }
      if (method === 'prompt.receipt') return { admission_id: params.admission_id, status: 'started', outcome: null }
      throw new Error(`unexpected method ${method}`)
    },
    composer: {
      submit(sessionId, text) {
        calls.push({ fn: 'composer.submit', sessionId, text })
        for (let i = 0; i < composerCopies; i++) history.push({ role: 'user', content: [{ type: 'text', text }] })
        return composerAck
      }
    },
    notify(note) {
      host.notes.push(note)
    }
  }
  return host
}

function makeCtx({ files = true } = {}) {
  const ctx = {
    registered: null,
    disposers: [],
    writes: [],
    clipboard: null,
    registerMany(contributions) {
      ctx.registered = contributions
      return () => {}
    },
    onDispose(fn) {
      ctx.disposers.push(fn)
    },
    os: {
      async writeClipboard(text) {
        ctx.clipboard = text
        return true
      }
    }
  }
  if (files) {
    ctx.files = {
      async writeText(folder, path, text) {
        ctx.writes.push({ folder, path, text })
        return { ok: true, path, size: text.length, mtimeMs: 0 }
      }
    }
  }
  return ctx
}

async function setup({ host = {}, globals = {}, ctx = {}, options = {} } = {}) {
  const plugin = await loadPlugin({ globals })
  const fakeHost = makeHost(host)
  const fakeCtx = makeCtx(ctx)
  let clock = Date.UTC(2026, 10, 20)
  const sleeps = []
  const probe = new plugin.ProbeA(fakeHost, fakeCtx, {
    now: () => clock,
    sleep: async ms => {
      sleeps.push(ms)
      clock += ms
    },
    ...options
  })
  return { plugin, host: fakeHost, ctx: fakeCtx, probe, sleeps }
}

// Values built inside the vm context have its own prototypes: compare them as plain JSON.
const same = (actual, expected) => assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected)
const rpc = host => host.calls.filter(c => c.fn !== 'profileRoutes')
const methods = host => host.calls.filter(c => c.fn === 'requestProfile').map(c => c.method)
const events = (probe, name) => probe.entries.filter(e => e.event === name)
const last = (probe, name) => events(probe, name).at(-1)

function tri(n, period, amplitude) {
  const t = Math.floor((n % period) * 4 * amplitude / period)
  if (t < amplitude) return t
  if (t < 3 * amplitude) return 2 * amplitude - t
  return t - 4 * amplitude
}

test('register only reads state, registers exactly the eight palette commands and logs focus changes', async () => {
  const host = makeHost()
  const plugin = await loadPlugin({ sdk: { host } })
  const ctx = makeCtx()
  plugin.default.register(ctx)
  assert.equal(plugin.default.id, 'sound-scout-probe-a')
  same(host.calls, [])
  const ids = ctx.registered.map(c => c.data.id)
  same(ids, Object.keys(LABELS).map(id => `sound-scout-probe-a.${id}`))
  for (const c of ctx.registered) {
    assert.equal(c.area, 'palette')
    assert.equal(c.data.label, LABELS[c.id])
    assert.equal(typeof c.data.run, 'function')
  }
  await new Promise(resolve => setTimeout(resolve, 0))
  const loaded = JSON.parse(ctx.writes.at(-1).text.trim().split('\n')[0])
  assert.equal(loaded.event, 'loaded')
  assert.equal(loaded.sdk.retainSession, 'undefined')
  assert.equal(loaded.sdk.requestProfile, 'function')
  assert.equal(loaded.sdk.desktopPluginsRoot, 'undefined')
  assert.equal(loaded.crypto_subtle, true)

  host.state.focusedStoredSessionId.set('stored-2')
  host.state.focusedSessionOwner.set(null)
  await new Promise(resolve => setTimeout(resolve, 0))
  const log = ctx.writes.at(-1).text.trim().split('\n').map(line => JSON.parse(line))
  const focus = log.filter(e => e.event === 'focus_change')
  same(focus.map(e => [e.stored, e.owner]), [
    ['stored-2', { connectionId: 'conn-1', profile: 'default' }],
    ['stored-2', null]
  ])
  assert.equal(ctx.disposers.length, 2)
  for (const dispose of ctx.disposers) dispose()
  assert.equal(host.state.focusedStoredSessionId.listeners.size, 0)
  assert.equal(host.state.focusedSessionOwner.listeners.size, 0)
  same(rpc(host), [])
})

test('snapshot makes no RPC and logs owner, routes and the redacted desktop-plugins directory', async () => {
  const { host, probe, ctx } = await setup({
    host: { routes: [ROUTE, OTHER_ROUTE] },
    globals: { hermesDesktop: { desktopPluginsRoot: async () => 'C:\\Users\\someone\\.hermes\\desktop-plugins' } }
  })
  await probe.run('snapshot-old')
  same(rpc(host), [])
  const s = last(probe, 'snapshot')
  assert.equal(s.case, 'old')
  assert.equal(s.stored, 'stored-1')
  assert.equal(s.runtime, 'runtime-1')
  same(s.owner, { connectionId: 'conn-1', profile: 'default' })
  assert.equal(s.profile, 'default')
  assert.equal(s.active_profile, 'default')
  assert.equal(s.connection, 'conn-1')
  assert.equal(s.gateway, 'open')
  same(s.routes, [
    { connectionId: 'conn-1', mode: 'remote', primary: null, profile: 'default', targetProfile: 'default' },
    { connectionId: 'conn-2', mode: 'local', primary: true, profile: 'default', targetProfile: 'default' }
  ])
  assert.equal(s.owner_matches, 1)
  assert.equal(s.plugins_root, '%USERPROFILE%\\.hermes\\desktop-plugins')

  host.state.focusedSessionOwner.set(null)
  await probe.run('snapshot-renamed')
  assert.equal(last(probe, 'snapshot').owner, null)
  assert.equal(last(probe, 'snapshot').owner_matches, 0)
  await probe.writes
  const text = ctx.writes.at(-1).text
  assert.equal(ctx.writes.at(-1).folder, 'data')
  assert.equal(ctx.writes.at(-1).path, 'probe-a-log.jsonl')
  assert.ok(!text.includes(RAW_CONN) && !text.includes('laptop-local') && !text.includes('someone'))
})

test('snapshot without the hermesDesktop bridge marks the directory unavailable', async () => {
  const { probe } = await setup()
  await probe.run('snapshot-new')
  same(Object.keys(last(probe, 'snapshot').plugins_root), ['unavailable'])
})

test('redactPath hides the user name of Windows and Linux paths', async () => {
  const { redactPath } = await loadPlugin()
  assert.equal(redactPath('C:\\Users\\someone\\.hermes\\desktop-plugins'), '%USERPROFILE%\\.hermes\\desktop-plugins')
  assert.equal(redactPath('/home/someone/hermes-home/attachments/x.wav'), '~/hermes-home/attachments/x.wav')
  assert.equal(redactPath('/root/.hermes/x'), '~/.hermes/x')
  assert.equal(redactPath('/srv/hermes/x'), '/srv/hermes/x')
  assert.equal(redactPath(null), null)
})

test('write commands refuse without a target and make no RPC', async () => {
  const { host, probe } = await setup()
  for (const id of WRITE_COMMANDS) await probe.run(id)
  same(host.calls, [])
  same(events(probe, 'refused').map(e => [e.command, e.reason]), WRITE_COMMANDS.map(id => [id, 'no_target']))
})

test('no chat write without a test button: every palette command but the three write ones makes no write call', async () => {
  const host = makeHost()
  const plugin = await loadPlugin({
    sdk: { host },
    globals: { hermesDesktop: { desktopPluginsRoot: async () => 'C:\\Users\\someone\\.hermes\\desktop-plugins' } }
  })
  const ctx = makeCtx()
  plugin.default.register(ctx)
  const readOnly = ctx.registered.filter(c => !WRITE_COMMANDS.includes(c.id))
  same(readOnly.map(c => c.id), ['snapshot-old', 'snapshot-renamed', 'snapshot-new', 'mark-target', 'copy-log'])
  for (const round of [1, 2]) {
    for (const c of readOnly) {
      c.data.run()
      await new Promise(resolve => setTimeout(resolve, 0))
    }
    host.state.focusedStoredSessionId.set(`stored-${round + 1}`)
  }
  await new Promise(resolve => setTimeout(resolve, 0))
  same(rpc(host), [])
  same(host.history.length, 2)
  const log = ctx.writes.at(-1).text.trim().split('\n').map(line => JSON.parse(line))
  assert.equal(log.filter(e => e.event === 'snapshot').length, 6)
  assert.ok(log.some(e => e.event === 'target_marked'))
})

test('marking a target needs an open chat', async () => {
  const { host, probe } = await setup()
  host.state.focusedStoredSessionId.set(null)
  await probe.run('mark-target')
  assert.equal(probe.target, null)
  assert.equal(last(probe, 'refused').reason, 'no_chat')
})

for (const [name, routes] of [['no route', []], ['two routes', [ROUTE, { ...ROUTE, mode: 'local' }]]]) {
  test(`route choice refuses ambiguous_route with ${name} and does not retain`, async () => {
    const { host, probe } = await setup({ host: { routes } })
    await probe.run('mark-target')
    await probe.run('attach')
    assert.equal(host.calls.filter(c => c.fn === 'retainProfile').length, 0)
    same([last(probe, 'refused').reason, last(probe, 'refused').matches], ['ambiguous_route', routes.length])
  })
}

test('route choice rereads a null owner while the target is on screen', async () => {
  const { host, probe } = await setup({ host: { owner: null } })
  await probe.run('mark-target')
  host.state.focusedSessionOwner.set({ connectionId: RAW_CONN, profile: 'default' })
  await probe.run('attach')
  assert.equal(last(probe, 'attach').owner_fallback, 'reread')
  assert.equal(host.calls.find(c => c.fn === 'retainProfile').route, ROUTE)
})

test('route choice falls back to the single route when the owner stays null', async () => {
  const { host, probe } = await setup({ host: { owner: null } })
  await probe.run('mark-target')
  host.state.focusedStoredSessionId.set('stored-2')
  host.state.focusedSessionOwner.set({ connectionId: RAW_CONN, profile: 'default' })
  await probe.run('attach')
  const entry = last(probe, 'attach')
  assert.equal(entry.owner_fallback, 'single_route')
  assert.equal(entry.visible, false)
})

test('attach retains, resumes and attaches the synthetic WAV, then releases once', async () => {
  const { host, probe, ctx } = await setup()
  await probe.run('mark-target')
  await probe.run('attach')
  const steps = rpc(host)
  same(steps.map(c => c.method ?? c.fn), ['retainProfile', 'session.resume', 'file.attach'])
  same(steps[0].options, { spawnPriority: 'foreground' })
  same(steps[1].params, { session_id: 'stored-1', omit_messages: true })
  const attach = steps[2]
  same(Object.keys(attach.params).sort(), ['data_url', 'name', 'session_id'])
  assert.equal(attach.params.session_id, 'runtime-1')
  assert.match(attach.params.name, /^sound-scout-probe-a-[0-9a-f]{12}\.wav$/)
  assert.equal(attach.timeout, 120000)
  assert.ok(attach.params.data_url.startsWith('data:audio/wav;base64,'))
  const bytes = Buffer.from(attach.params.data_url.split(',')[1], 'base64')
  assert.equal(bytes.length, WAV_BYTES)
  assert.equal(bytes.toString('latin1', 0, 4), 'RIFF')
  assert.equal(bytes.readUInt16LE(20), 1)
  assert.equal(bytes.readUInt16LE(22), 2)
  assert.equal(bytes.readUInt32LE(24), 48000)
  assert.equal(bytes.readUInt16LE(34), 16)
  const entry = last(probe, 'attach')
  assert.equal(entry.sha256, createHash('sha256').update(bytes).digest('hex'))
  assert.equal(entry.bytes, WAV_BYTES)
  assert.equal(entry.visible, true)
  assert.equal(entry.runtime_session, 'runtime-1')
  same(entry.route, { conn: 'conn-1', profile: 'default' })
  assert.equal(entry.path, `~/hermes-home/attachments/${attach.params.name}`)
  assert.equal(entry.ref_path, entry.path)
  same(Object.keys(entry.ms).sort(), ['attach', 'encode', 'resume', 'retain'])
  assert.equal(host.released, 1)
  assert.equal(probe.lastAttach.sha256, entry.sha256)
  await probe.writes
  assert.ok(!ctx.writes.at(-1).text.includes('/home/someone'))
})

for (const [name, hostOptions, reason] of [
  ['a resume error', { fail: { 'session.resume': 'socket closed' } }, null],
  ['a file.attach error', { fail: { 'file.attach': 'permission_denied' } }, null],
  ['an answer without attached', { attachReply: { attached: false, ref_text: '' } }, 'attach_unconfirmed'],
  ['a resume without session_id', { fail: {}, attachReply: null }, 'resume_failed']
]) {
  test(`attach releases the profile exactly once after ${name}`, async () => {
    const { host, probe } = await setup({ host: hostOptions })
    if (reason === 'resume_failed') {
      const request = host.requestProfile
      host.requestProfile = async (...args) => (args[1] === 'session.resume' ? {} : request(...args))
    }
    await probe.run('mark-target')
    await probe.run('attach')
    assert.equal(host.released, 1)
    assert.equal(events(probe, 'attach').length, 0)
    if (reason) assert.equal(last(probe, 'refused').reason, reason)
    else assert.equal(last(probe, 'error').command, 'attach')
    assert.equal(host.notes.at(-1).kind, reason ? 'warning' : 'error')
  })
}

test('attach without crypto.subtle logs a null sha256 with a reason', async () => {
  const { probe } = await setup({ globals: { crypto: { getRandomValues: a => globalThis.crypto.getRandomValues(a) } } })
  await probe.run('mark-target')
  await probe.run('attach')
  const entry = last(probe, 'attach')
  assert.equal(entry.sha256, null)
  assert.equal(entry.sha256_reason, 'crypto_subtle_unavailable')
})

test('a write command while another runs is refused as busy', async () => {
  let open
  const gate = new Promise(resolve => { open = resolve })
  const { host, probe } = await setup({ host: { retainGate: gate } })
  await probe.run('mark-target')
  const first = probe.run('attach')
  await new Promise(resolve => setTimeout(resolve, 0))
  await probe.run('prompt-submit')
  assert.equal(last(probe, 'refused').reason, 'busy')
  open()
  await first
  assert.equal(events(probe, 'attach').length, 1)
  assert.equal(host.released, 1)
})

async function submitWith(command, hostOptions = {}, focus = 'stored-1') {
  const env = await setup({ host: hostOptions })
  await env.probe.run('mark-target')
  env.host.state.focusedStoredSessionId.set(focus)
  await env.probe.run(command)
  return { ...env, entry: last(env.probe, 'submit') }
}

test('prompt.submit sends exactly three params, replays the submission_id and reads the receipt', async () => {
  const { host, entry, sleeps, ctx, probe } = await submitWith('prompt-submit')
  const submits = host.calls.filter(c => c.method === 'prompt.submit')
  assert.equal(submits.length, 2)
  same(Object.keys(submits[0].params).sort(), ['session_id', 'submission_id', 'text'])
  same(submits[1].params, JSON.parse(JSON.stringify(submits[0].params)))
  assert.equal(submits[0].params.session_id, 'runtime-1')
  assert.equal(submits[0].params.submission_id, `sound-scout-probe-a-${entry.token}`)
  assert.ok(submits[0].params.text.startsWith(`sound-scout-probe-a ${entry.token} (prompt.submit, visible): `))
  assert.ok(!submits[0].params.text.includes('@file:'))
  same(host.calls.find(c => c.method === 'prompt.receipt').params, { session_id: 'runtime-1', admission_id: 'adm-1' })
  same(entry.ack, { admission_id: 'adm-1', status: 'queued', outcome: null })
  assert.equal(entry.replay_same_admission, true)
  assert.equal(entry.receipt_after_status, 'started')
  assert.equal(entry.verdict, 'real')
  assert.equal(entry.occurrences, 1)
  assert.equal(entry.present_after_ms, 0)
  assert.equal(entry.message_count_before, 2)
  assert.equal(entry.message_count_after, 3)
  assert.equal(entry.visible, true)
  assert.equal(host.released, 1)
  const reads = host.calls.filter(c => c.method === 'session.resume' && !c.params.omit_messages)
  same(reads[0].params, { session_id: 'stored-1' })
  assert.equal(reads.length, 16)
  assert.equal(sleeps.reduce((a, b) => a + b, 0), 30000)
  await probe.writes
  const log = ctx.writes.at(-1).text
  assert.ok(!log.includes(PRIVATE_TEXT) && !log.includes('private answer') && !log.includes('тестовое сообщение'))
})

for (const [verdict, hostOptions] of [
  ['duplicate', { submitCopies: 2 }],
  ['false_ack', { submitCopies: 0 }],
  ['refused', { fail: { 'prompt.submit': 'invalid_params' } }],
  ['false_refusal', { failAfterWrite: 'timeout' }],
  ['inconclusive', { replayNewAdmission: true }]
]) {
  test(`prompt.submit verdict ${verdict}`, async () => {
    const { entry, host } = await submitWith('prompt-submit', hostOptions)
    assert.equal(entry.verdict, verdict)
    assert.equal(host.released, 1)
    if (verdict === 'refused') {
      assert.equal(entry.error, 'invalid_params')
      same(methods(host).filter(m => m.startsWith('prompt.')), ['prompt.submit'])
    }
  })
}

// A failed or empty history read is no evidence the message is missing: never false_ack or refused.
for (const command of ['prompt-submit', 'composer-submit']) {
  for (const [name, hostOptions] of [
    ['every history read fails', { failVerify: 'socket closed' }],
    ['history reads return no messages array', { verifyReply: { session_id: 'runtime-1' } }],
    ['history reads fail and the submit is refused', { failVerify: 'socket closed', fail: { 'prompt.submit': 'x' }, composerAck: false }]
  ]) {
    test(`${command} is inconclusive when ${name}`, async () => {
      const { entry, host } = await submitWith(command, hostOptions)
      assert.equal(entry.verdict, 'inconclusive')
      assert.equal(entry.occurrences, 0)
      assert.equal(entry.verify_valid_reads, 0)
      assert.equal(host.released, 1)
    })
  }
}

test('one failed history read with no marker makes the verdict inconclusive, a clean miss stays false_ack', async () => {
  for (const command of ['prompt-submit', 'composer-submit']) {
    const env = await setup({ host: { submitCopies: 0, composerCopies: 0 } })
    const request = env.host.requestProfile
    let reads = 0
    env.host.requestProfile = async (...args) => {
      if (args[1] === 'session.resume' && !args[2].omit_messages && reads++ === 3) throw new Error('socket closed')
      return request(...args)
    }
    await env.probe.run('mark-target')
    await env.probe.run(command)
    const entry = last(env.probe, 'submit')
    assert.equal(entry.verify_read_errors, 1)
    assert.equal(entry.verify_valid_reads, 15)
    assert.equal(entry.verdict, 'inconclusive')
    assert.equal((await submitWith(command, { submitCopies: 0, composerCopies: 0 })).entry.verdict, 'false_ack')
  }
})

test('mark-target while a write command runs is refused and the command keeps its target', async () => {
  let open
  const gate = new Promise(resolve => { open = resolve })
  const { host, probe } = await setup({ host: { retainGate: gate } })
  await probe.run('mark-target')
  const first = probe.run('prompt-submit')
  await new Promise(resolve => setTimeout(resolve, 0))
  host.state.focusedStoredSessionId.set('stored-2')
  await probe.run('mark-target')
  same([last(probe, 'refused').command, last(probe, 'refused').reason], ['mark-target', 'busy'])
  assert.equal(probe.target.stored, 'stored-1')
  open()
  await first
  const reads = host.calls.filter(c => c.method === 'session.resume')
  assert.ok(reads.length > 1 && reads.every(c => c.params.session_id === 'stored-1'))
  assert.equal(last(probe, 'submit').verdict, 'real')
})

test('prompt.submit to a target off screen is marked hidden', async () => {
  const { entry, host } = await submitWith('prompt-submit', {}, 'stored-2')
  assert.equal(entry.visible, false)
  assert.ok(host.calls.find(c => c.method === 'prompt.submit').params.text.includes('(prompt.submit, hidden)'))
})

for (const [verdict, hostOptions] of [
  ['real', {}],
  ['false_ack', { composerCopies: 0 }],
  ['false_refusal', { composerAck: false }],
  ['refused', { composerAck: false, composerCopies: 0 }],
  ['duplicate', { composerCopies: 2 }]
]) {
  test(`composer.submit verdict ${verdict}`, async () => {
    const { entry, host } = await submitWith('composer-submit', hostOptions)
    assert.equal(entry.method, 'composer.submit')
    assert.equal(entry.verdict, verdict)
    assert.equal(host.released, 1)
    const submit = host.calls.find(c => c.fn === 'composer.submit')
    assert.equal(submit.sessionId, 'stored-1')
    assert.ok(submit.text.startsWith(`sound-scout-probe-a ${entry.token} (composer.submit, visible): `))
    assert.ok(!methods(host).some(m => m.startsWith('prompt.')))
  })
}

test('composer.submit marks visible and hidden targets', async () => {
  assert.equal((await submitWith('composer-submit', {}, 'stored-1')).entry.visible, true)
  assert.equal((await submitWith('composer-submit', {}, 'stored-2')).entry.visible, false)
})

test('copy log puts the same journal on the clipboard as in ctx.files', async () => {
  const { probe, ctx } = await setup()
  await probe.run('snapshot-old')
  await probe.run('copy-log')
  await probe.writes
  assert.equal(ctx.clipboard, ctx.writes.at(-1).text)
  assert.equal(ctx.clipboard.trim().split('\n').length, probe.entries.length)
})

test('a failing log write is recorded once and does not stop the probe', async () => {
  const { probe, ctx } = await setup()
  ctx.files.writeText = async () => ({ ok: false, error: 'too_large' })
  await probe.run('snapshot-old')
  await probe.run('snapshot-new')
  await probe.writes
  assert.equal(events(probe, 'log_write_failed').length, 1)
  assert.equal(events(probe, 'snapshot').length, 2)
})

test('the synthetic WAV holds the integer triangles exactly', async () => {
  const { syntheticWav } = await loadPlugin()
  const bytes = Buffer.from(await syntheticWav())
  assert.equal(bytes.length, WAV_BYTES)
  for (let n = 0, offset = 44; n < 48000 * 30; n++, offset += 4) {
    if (bytes.readInt16LE(offset) !== tri(n, 100, 8000) || bytes.readInt16LE(offset + 2) !== tri(n, 150, 6000)) {
      assert.fail(`frame ${n} differs`)
    }
  }
})

test('the plugin source uses only the SDK and the four RPC methods', async () => {
  const source = await readFile(PROBE_PATH, 'utf8')
  for (const banned of [/\bfetch\(/, /XMLHttpRequest/, /WebSocket/, /retainSession\(/, /host\.request\(/, /\beval\b/]) {
    assert.doesNotMatch(source, banned)
  }
  const imports = [...source.matchAll(/^\s*import\b.*?from\s+['"]([^'"]+)['"]/gm)].map(m => m[1])
  same(imports, ['@hermes/plugin-sdk'])
  assert.doesNotMatch(source, /\bimport\s*\(/)
  const rpc = new Set([...source.matchAll(/'((?:session|file|prompt)\.[a-z_]+)'/g)].map(m => m[1]))
  same([...rpc].sort(), ['file.attach', 'prompt.receipt', 'prompt.submit', 'session.resume'])
})
