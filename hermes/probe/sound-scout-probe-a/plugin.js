// Spike A diagnostic Desktop plugin (#95, decision 0016). Palette commands only: the snapshots and the
// focus log read state; the attach and submit commands write only into the chat the owner marked as
// the test target. The log is JSON Lines with paths and connection ids redacted.
import { host, PALETTE_AREA } from '@hermes/plugin-sdk'

export const PROBE_ID = 'sound-scout-probe-a'
const TITLE = 'Sound Scout probe A'
const RATE = 48000
const SECONDS = 30
const CHUNK_FRAMES = 4096
const LOG_FILE = 'probe-a-log.jsonl'
const ERROR_CHARS = 300
const ATTACH_TIMEOUT_MS = 120000
const FOREGROUND = { spawnPriority: 'foreground' }

// Ported unchanged from renidere/hermes-voice-note v0.4.2 (ced69ea, plugin.js), MIT License.
export function encodeWav(chunks, sampleRate) {
  const channels = chunks[0]?.length;
  if (![1, 2].includes(channels) || !Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 192000) {
    throw new Error('Некорректный формат записи');
  }
  let frames = 0;
  for (const chunk of chunks) {
    if (chunk.length !== channels || !chunk[0].length || chunk.some(channel => channel.length !== chunk[0].length)) {
      throw new Error('Некорректные каналы записи');
    }
    frames += chunk[0].length;
  }
  const buffer = new ArrayBuffer(44 + frames * channels * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true); view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, buffer.byteLength - 44, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (let frame = 0; frame < chunk[0].length; frame++) {
      for (let channel = 0; channel < channels; channel++) {
        const sample = chunk[channel][frame];
        if (!Number.isFinite(sample)) throw new Error('Некорректные отсчёты записи');
        const value = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
        offset += 2;
      }
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

// The same integer triangle scripts/probe_a_check_attach.py generates.
function tri(n, period, amplitude) {
  const t = Math.floor((n % period) * 4 * amplitude / period)
  if (t < amplitude) return t
  if (t < 3 * amplitude) return 2 * amplitude - t
  return t - 4 * amplitude
}

const toFloat = v => v / (v < 0 ? 32768 : 32767)

export function syntheticChunks() {
  const frames = RATE * SECONDS
  const chunks = []
  for (let start = 0; start < frames; start += CHUNK_FRAMES) {
    const size = Math.min(CHUNK_FRAMES, frames - start)
    const left = new Float32Array(size)
    const right = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      left[i] = toFloat(tri(start + i, 100, 8000))
      right[i] = toFloat(tri(start + i, 150, 6000))
    }
    chunks.push([left, right])
  }
  return chunks
}

export async function syntheticWav() {
  return new Uint8Array(await encodeWav(syntheticChunks(), RATE).arrayBuffer())
}

export function redactPath(s) {
  if (typeof s !== 'string') return s
  return s
    .replace(/^[A-Za-z]:[\\/]Users[\\/][^\\/]+/, '%USERPROFILE%')
    .replace(/^\/home\/[^/]+/, '~')
    .replace(/^\/root(?=\/|$)/, '~')
}

function redactText(s) {
  return s
    .replace(/[A-Za-z]:[\\/]Users[\\/][^\\/\s'"]+/g, '%USERPROFILE%')
    .replace(/\/home\/[^/\s'"]+/g, '~')
    .replace(/(^|[\s'"])\/root(?=\/|\s|$)/g, '$1~')
}

class Refusal extends Error {
  constructor(code, details = {}) {
    super(code)
    this.code = code
    this.details = details
  }
}

function randomToken() {
  const bytes = new Uint8Array(6)
  if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(bytes)
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256)
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
}

function messageText(message) {
  const content = message?.content ?? message?.text
  if (typeof content === 'string') return content
  if (Array.isArray(content)) return content.map(part => (typeof part === 'string' ? part : part?.text ?? '')).join('')
  return ''
}

function countMarker(messages, marker) {
  if (!Array.isArray(messages)) return 0
  return messages.filter(m => m?.role === 'user' && messageText(m).includes(marker)).length
}

// A marker absent from history proves nothing when a history read failed or none returned messages.
function unverified(seen) {
  return seen.occurrences === 0 && (seen.valid_reads === 0 || seen.read_errors > 0)
}

function readDataUrl(bytes) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(new Error('data_url_failed'))
    reader.readAsDataURL(new Blob([bytes], { type: 'audio/wav' }))
  })
}

async function sha256Hex(bytes) {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) return { sha256: null, sha256_reason: 'crypto_subtle_unavailable' }
  const digest = new Uint8Array(await subtle.digest('SHA-256', bytes))
  return { sha256: Array.from(digest, b => b.toString(16).padStart(2, '0')).join('') }
}

const COMMANDS = [
  { id: 'snapshot-old', label: 'Probe A: snapshot — old chat', run: p => p.snapshot('old') },
  { id: 'snapshot-renamed', label: 'Probe A: snapshot — renamed chat', run: p => p.snapshot('renamed') },
  { id: 'snapshot-new', label: 'Probe A: snapshot — new chat', run: p => p.snapshot('new') },
  { id: 'mark-target', label: 'Probe A: mark this chat as test target', run: p => p.markTarget() },
  { id: 'attach', label: 'Probe A: attach synthetic 30 s WAV to target', write: true, run: p => p.attach() },
  { id: 'prompt-submit', label: 'Probe A: prompt.submit to target', write: true, run: p => p.promptSubmit() },
  { id: 'composer-submit', label: 'Probe A: composer.submit to target', write: true, run: p => p.composerSubmit() },
  { id: 'copy-log', label: 'Probe A: copy log', run: p => p.copyLog() }
]

export class ProbeA {
  constructor(host, ctx, {
    sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    now = () => Date.now(),
    verifyTimeoutMs = 30000,
    verifyEveryMs = 2000
  } = {}) {
    this.host = host
    this.ctx = ctx
    this.sleep = sleep
    this.now = now
    this.verifyTimeoutMs = verifyTimeoutMs
    this.verifyEveryMs = verifyEveryMs
    this.entries = []
    this.aliases = new Map()
    this.target = null
    this.busy = false
    this.lastAttach = null
    this.lastFocus = null
    this.logWriteFailed = false
    this.writes = Promise.resolve()
  }

  static get commands() {
    return COMMANDS.map(({ id, label }) => ({ id, label }))
  }

  alias(connectionId) {
    if (connectionId === null || connectionId === undefined || connectionId === '') return connectionId ?? null
    if (!this.aliases.has(connectionId)) this.aliases.set(connectionId, `conn-${this.aliases.size + 1}`)
    return this.aliases.get(connectionId)
  }

  scrub(error) {
    let text = String(error?.message ?? error)
    for (const [raw, alias] of this.aliases) text = text.split(raw).join(alias)
    return redactText(text).slice(0, ERROR_CHARS)
  }

  owner(value) {
    return value ? { connectionId: this.alias(value.connectionId), profile: value.profile } : null
  }

  stored() {
    return this.host.state.focusedStoredSessionId.get()
  }

  record(event, fields = {}) {
    this.entries.push({ seq: this.entries.length + 1, at: new Date(this.now()).toISOString(), event, ...fields })
    this.flush()
  }

  logText() {
    return this.entries.map(entry => `${JSON.stringify(entry)}\n`).join('')
  }

  flush() {
    const files = this.ctx.files
    if (!files) return
    this.writes = this.writes.then(async () => {
      try {
        const result = await files.writeText('data', LOG_FILE, this.logText())
        if (result && result.ok === false) throw new Error(result.error || 'write_failed')
      } catch (error) {
        if (this.logWriteFailed) return
        this.logWriteFailed = true
        this.entries.push({
          seq: this.entries.length + 1,
          at: new Date(this.now()).toISOString(),
          event: 'log_write_failed',
          error: this.scrub(error)
        })
      }
    })
  }

  notify(message, kind = 'info') {
    this.host.notify({ kind, title: TITLE, message })
  }

  start() {
    const h = this.host
    this.record('loaded', {
      sdk: {
        profileRoutes: typeof h.profileRoutes,
        retainProfile: typeof h.retainProfile,
        requestProfile: typeof h.requestProfile,
        composer_submit: typeof h.composer?.submit,
        retainSession: typeof h.retainSession,
        files: typeof this.ctx.files,
        writeClipboard: typeof this.ctx.os?.writeClipboard,
        desktopPluginsRoot: typeof globalThis.hermesDesktop?.desktopPluginsRoot
      },
      crypto_subtle: Boolean(globalThis.crypto?.subtle),
      stored: this.stored(),
      owner: this.owner(h.state.focusedSessionOwner.get())
    })
    const onFocus = () => {
      const stored = this.stored()
      const owner = this.owner(h.state.focusedSessionOwner.get())
      const key = JSON.stringify([stored, owner])
      if (key === this.lastFocus) return
      this.lastFocus = key
      this.record('focus_change', { stored, owner })
    }
    this.lastFocus = JSON.stringify([this.stored(), this.owner(h.state.focusedSessionOwner.get())])
    for (const atom of [h.state.focusedStoredSessionId, h.state.focusedSessionOwner]) {
      this.ctx.onDispose(atom.listen(onFocus))
    }
  }

  async run(id) {
    const command = COMMANDS.find(c => c.id === id)
    if (!command) throw new Error(`unknown command ${id}`)
    if (command.write) {
      if (!this.target) return this.refuse(id, 'no_target', 'Сначала отметь тестовый чат')
      if (this.busy) return this.refuse(id, 'busy', 'Предыдущая команда ещё работает')
      this.busy = true
    }
    try {
      await command.run(this)
    } catch (error) {
      if (error instanceof Refusal) {
        this.record('refused', { command: id, reason: error.code, ...error.details })
        this.notify(`${id}: отказ ${error.code}`, 'warning')
      } else {
        this.record('error', { command: id, step: error?.step ?? null, error: this.scrub(error) })
        this.notify(`${id}: ошибка ${this.scrub(error)}`, 'error')
      }
    } finally {
      if (command.write) this.busy = false
    }
  }

  refuse(command, reason, message) {
    this.record('refused', { command, reason })
    this.notify(`${command}: ${message}`, 'warning')
  }

  async snapshot(kind) {
    const s = this.host.state
    const owner = s.focusedSessionOwner.get()
    const routes = await this.host.profileRoutes()
    let pluginsRoot
    const desktop = globalThis.hermesDesktop
    if (typeof desktop?.desktopPluginsRoot !== 'function') {
      pluginsRoot = { unavailable: 'no_hermesDesktop_desktopPluginsRoot' }
    } else {
      try {
        pluginsRoot = redactPath(await desktop.desktopPluginsRoot())
      } catch (error) {
        pluginsRoot = { unavailable: this.scrub(error) }
      }
    }
    const ownerMatches = owner
      ? routes.filter(r => r.connectionId === owner.connectionId && r.profile === owner.profile).length
      : 0
    this.record('snapshot', {
      case: kind,
      stored: s.focusedStoredSessionId.get(),
      runtime: s.focusedSessionId.get(),
      owner: this.owner(owner),
      profile: s.focusedSessionProfile.get(),
      active_profile: s.profile.get(),
      connection: this.alias(s.connectionId.get()),
      gateway: s.gateway.get(),
      routes: routes.map(r => ({
        connectionId: this.alias(r.connectionId),
        mode: r.mode,
        primary: r.primary ?? null,
        profile: r.profile,
        targetProfile: r.targetProfile
      })),
      owner_matches: ownerMatches,
      plugins_root: pluginsRoot
    })
    this.notify(`snapshot ${kind}: routes=${routes.length} owner_matches=${ownerMatches}`)
  }

  markTarget() {
    if (this.busy) return this.refuse('mark-target', 'busy', 'Дождись конца текущей команды')
    const stored = this.stored()
    if (!stored) {
      this.record('refused', { command: 'mark-target', reason: 'no_chat' })
      this.notify('Открой чат, который станет тестовым', 'warning')
      return
    }
    this.target = { stored, owner: this.host.state.focusedSessionOwner.get() }
    this.record('target_marked', { stored, owner: this.owner(this.target.owner) })
    this.notify(`Тестовый чат: ${stored}`)
  }

  async pickRoute() {
    let owner = this.target.owner
    let ownerFallback = null
    if (!owner && this.stored() === this.target.stored) {
      owner = this.host.state.focusedSessionOwner.get()
      if (owner) ownerFallback = 'reread'
    }
    const routes = await this.host.profileRoutes()
    if (owner) {
      const matches = routes.filter(r => r.connectionId === owner.connectionId && r.profile === owner.profile)
      if (matches.length === 1) return { route: matches[0], ownerFallback }
      throw new Refusal('ambiguous_route', { matches: matches.length })
    }
    if (routes.length === 1) return { route: routes[0], ownerFallback: 'single_route' }
    throw new Refusal('ambiguous_route', { matches: routes.length })
  }

  async step(name, timings, fn) {
    const started = this.now()
    try {
      return await fn()
    } catch (error) {
      if (error && typeof error === 'object' && !error.step) error.step = name
      throw error
    } finally {
      timings[name] = this.now() - started
    }
  }

  async resume(route, ms) {
    const resumed = await this.step('resume', ms, () =>
      this.host.requestProfile(
        route,
        'session.resume',
        { session_id: this.target.stored, omit_messages: true },
        undefined,
        FOREGROUND
      ))
    if (typeof resumed?.session_id !== 'string' || !resumed.session_id) throw new Refusal('resume_failed')
    return resumed
  }

  async attach() {
    const visible = this.stored() === this.target.stored
    const { route, ownerFallback } = await this.pickRoute()
    const ms = {}
    let release
    try {
      release = await this.step('retain', ms, () => this.host.retainProfile(route, FOREGROUND))
      const resumed = await this.resume(route, ms)
      const { bytes, hash, dataUrl } = await this.step('encode', ms, async () => {
        const bytes = await syntheticWav()
        return { bytes, hash: await sha256Hex(bytes), dataUrl: await readDataUrl(bytes) }
      })
      const name = `${PROBE_ID}-${randomToken()}.wav`
      const uploaded = await this.step('attach', ms, () =>
        this.host.requestProfile(
          route,
          'file.attach',
          { session_id: resumed.session_id, name, data_url: dataUrl },
          ATTACH_TIMEOUT_MS
        ))
      const confirmed = uploaded?.attached === true &&
        typeof uploaded.ref_text === 'string' && uploaded.ref_text.startsWith('@file:')
      const entry = {
        visible,
        owner_fallback: ownerFallback,
        route: { conn: this.alias(route.connectionId), profile: route.profile },
        runtime_session: resumed.session_id,
        name: uploaded?.name ?? null,
        path: redactPath(uploaded?.path ?? null),
        ref_path: redactPath(uploaded?.ref_path ?? null),
        attached: uploaded?.attached ?? null,
        uploaded: uploaded?.uploaded ?? null,
        bytes: bytes.length,
        ...hash,
        ms
      }
      if (!confirmed) throw new Refusal('attach_unconfirmed', entry)
      this.lastAttach = entry
      this.record('attach', entry)
      this.notify(`attach ${visible ? 'visible' : 'hidden'}: ${entry.path} sha256=${entry.sha256}`)
    } finally {
      release?.()
    }
  }

  submitText(marker, method, visible) {
    return `${marker} (${method}, ${visible ? 'visible' : 'hidden'}): тестовое сообщение зонда Sound Scout, ` +
      'ничего не делай, ответь одним словом «ок».'
  }

  async verify(route, marker) {
    const started = this.now()
    const seen = {
      occurrences: 0, present_after_ms: null, message_count_after: null, reads: 0, valid_reads: 0, read_errors: 0
    }
    for (;;) {
      try {
        const answer = await this.host.requestProfile(
          route, 'session.resume', { session_id: this.target.stored }, undefined, FOREGROUND)
        seen.reads++
        if (Array.isArray(answer?.messages)) seen.valid_reads++
        const found = countMarker(answer?.messages, marker)
        if (found > 0 && seen.present_after_ms === null) seen.present_after_ms = this.now() - started
        seen.occurrences = Math.max(seen.occurrences, found)
        seen.message_count_after = answer?.message_count ?? null
      } catch (error) {
        seen.read_errors++
        seen.last_read_error = this.scrub(error)
      }
      if (this.now() - started >= this.verifyTimeoutMs) return seen
      await this.sleep(this.verifyEveryMs)
    }
  }

  async promptSubmit() {
    const visible = this.stored() === this.target.stored
    const { route, ownerFallback } = await this.pickRoute()
    const token = randomToken()
    const marker = `${PROBE_ID} ${token}`
    const ms = {}
    let release
    try {
      release = await this.step('retain', ms, () => this.host.retainProfile(route, FOREGROUND))
      const resumed = await this.resume(route, ms)
      const params = {
        session_id: resumed.session_id,
        text: this.submitText(marker, 'prompt.submit', visible),
        submission_id: `${PROBE_ID}-${token}`
      }
      const result = { ack: null, replay_same_admission: false, replay_status: null, receipt_after_status: null }
      let submitError = null
      try {
        const ack = await this.host.requestProfile(route, 'prompt.submit', params)
        result.ack = { admission_id: ack?.admission_id ?? null, status: ack?.status ?? null, outcome: ack?.outcome ?? null }
      } catch (error) {
        submitError = this.scrub(error)
      }
      if (result.ack) {
        try {
          const replay = await this.host.requestProfile(route, 'prompt.submit', { ...params })
          result.replay_same_admission = Boolean(result.ack.admission_id) && replay?.admission_id === result.ack.admission_id
          result.replay_status = replay?.status ?? null
        } catch (error) {
          result.replay_error = this.scrub(error)
        }
        try {
          const after = await this.host.requestProfile(route, 'prompt.receipt',
            { session_id: resumed.session_id, admission_id: result.ack.admission_id })
          result.receipt_after_status = after?.status ?? null
        } catch (error) {
          result.receipt_error = this.scrub(error)
        }
      }
      const seen = await this.step('verify', ms, () => this.verify(route, marker))
      const n = seen.occurrences
      const admitted = Boolean(result.ack?.admission_id)
      let verdict = 'inconclusive'
      if (n > 1) verdict = 'duplicate'
      else if (unverified(seen)) verdict = 'inconclusive'
      else if (submitError && n === 0) verdict = 'refused'
      else if (submitError) verdict = 'false_refusal'
      else if (admitted && result.replay_same_admission && n === 1) verdict = 'real'
      else if (admitted && n === 0) verdict = 'false_ack'
      this.record('submit', {
        method: 'prompt.submit',
        visible,
        owner_fallback: ownerFallback,
        token,
        ...result,
        error: submitError,
        occurrences: n,
        present_after_ms: seen.present_after_ms,
        message_count_before: resumed.message_count ?? null,
        message_count_after: seen.message_count_after,
        verify_reads: seen.reads,
        verify_valid_reads: seen.valid_reads,
        verify_read_errors: seen.read_errors,
        verdict,
        ms
      })
      this.notify(`prompt.submit ${visible ? 'visible' : 'hidden'}: ${verdict}`, verdict === 'real' ? 'success' : 'warning')
    } finally {
      release?.()
    }
  }

  async composerSubmit() {
    const visible = this.stored() === this.target.stored
    const { route, ownerFallback } = await this.pickRoute()
    const token = randomToken()
    const marker = `${PROBE_ID} ${token}`
    const ms = {}
    let release
    try {
      release = await this.step('retain', ms, () => this.host.retainProfile(route, FOREGROUND))
      const resumed = await this.resume(route, ms)
      const ack = this.host.composer.submit(this.target.stored, this.submitText(marker, 'composer.submit', visible))
      const seen = await this.step('verify', ms, () => this.verify(route, marker))
      const n = seen.occurrences
      let verdict
      if (n > 1) verdict = 'duplicate'
      else if (unverified(seen)) verdict = 'inconclusive'
      else if (ack) verdict = n === 1 ? 'real' : 'false_ack'
      else verdict = n >= 1 ? 'false_refusal' : 'refused'
      this.record('submit', {
        method: 'composer.submit',
        visible,
        owner_fallback: ownerFallback,
        token,
        ack: ack === true,
        occurrences: n,
        present_after_ms: seen.present_after_ms,
        message_count_before: resumed.message_count ?? null,
        message_count_after: seen.message_count_after,
        verify_reads: seen.reads,
        verify_valid_reads: seen.valid_reads,
        verify_read_errors: seen.read_errors,
        verdict,
        ms
      })
      this.notify(`composer.submit ${visible ? 'visible' : 'hidden'}: ${verdict}`, verdict === 'real' ? 'success' : 'warning')
    } finally {
      release?.()
    }
  }

  async copyLog() {
    const writeClipboard = this.ctx.os?.writeClipboard
    if (typeof writeClipboard !== 'function') {
      this.notify('Буфер обмена недоступен', 'warning')
      return
    }
    const copied = await writeClipboard(this.logText())
    this.notify(copied ? `Журнал скопирован: ${this.entries.length} записей` : 'Не удалось скопировать журнал',
      copied ? 'success' : 'warning')
  }
}

export default {
  id: PROBE_ID,
  name: TITLE,
  register(ctx) {
    const probe = new ProbeA(host, ctx)
    probe.start()
    ctx.registerMany(COMMANDS.map(({ id, label }) => ({
      id,
      area: PALETTE_AREA,
      data: { id: `${PROBE_ID}.${id}`, label, keywords: ['sound', 'scout', 'probe'], run: () => void probe.run(id) }
    })))
  }
}
