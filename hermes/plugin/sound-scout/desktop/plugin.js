// Desktop half of the Sound Scout package (#101, decision 0013 § 3.2–3.4): the recorder of the
// composer (#103), the job card above the input of the job's chat (#108) and the health command of
// the palette. The recorder sends by the stock path Spike A chose (decision 0016). The card follows
// the job by polling GET /jobs/{job_id} and shows its report as plain text: it holds the transcript,
// untrusted data, so it is never rendered as Markdown. "Прослушать" reads the sent WAV from the station
// again at each open (GET /recordings/{name}) (#214) and plays it in one detached new Audio() driven by
// SDK buttons, as the stock radio plugin of Desktop 0.23.3 does, so no render touches playback (#216).
import { atom, host, Button, Codicon, COMPOSER_AREAS, Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, PALETTE_AREA, useComposerSession, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const JOB_ID = /^[0-9a-f]{32}$/
const RECORDING_NAME = /^sound-scout-[0-9a-f]{12}\.wav$/
const DATA_URL_PREFIX = 'data:audio/wav;base64,'
const RECORDING_TIMEOUT_MS = 120000
// Ссылка blob: живёт после закрытия плеера, чтобы начатое скачивание успело забрать байты.
const URL_RELEASE_MS = 30000
const FOREGROUND = { spawnPriority: 'foreground' }
const ATTACH_TIMEOUT_MS = 120000
const UPLOAD_TIMEOUT_MS = 120000
const STATUS_TIMEOUT_MS = 10000
const HEALTH_TIMEOUT_MS = 10000
const SEEK_STEP_S = 5
const PLAYER_EVENTS = ['loadedmetadata', 'durationchange', 'timeupdate', 'play', 'pause', 'ended', 'seeked']
const MEDIA_ERRORS = { 1: 'MEDIA_ERR_ABORTED', 2: 'MEDIA_ERR_NETWORK', 3: 'MEDIA_ERR_DECODE', 4: 'MEDIA_ERR_SRC_NOT_SUPPORTED' }
// Границы версии R02: лимит записи, когда `GET /health` не дал бюджетов (#174).
const R02_BUDGET = { max_duration_seconds: 180, max_input_bytes: 104857600 }
const WAV_HEADER_BYTES = 44

function describeHealth(answer) {
  // An answer without a version string says so; the line never shows a guessed version.
  const version = typeof answer.version === 'string' && answer.version ? answer.version : 'недоступна'
  const budgets = answer.budgets
    ? `до ${answer.budgets.max_duration_seconds} с, до ${answer.budgets.max_input_bytes} байт`
    : 'бюджеты недоступны'
  const intake = answer.intake.ready
    ? 'приём записей готов'
    : `приём записей: ${answer.intake.error}: ${answer.intake.detail}`
  return `version=${version} pid=${answer.pid} session_owner=${answer.session_owner}; ${budgets}; ${intake}`
}

// From renidere/hermes-voice-note v0.4.2 ced69ea, MIT.
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

class RecorderError extends Error {
  constructor(code, message = code) {
    super(message)
    this.code = code
  }
}

function readDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(new Error('Не удалось прочитать запись'))
    reader.readAsDataURL(blob)
  })
}

// Desktop отдаёт сбой ctx.rest как ошибку IPC: снимаем обёртку и оставляем его собственный текст.
function desktopErrorText(error) {
  const text = String(error?.message ?? error ?? '').replace(/\s+/g, ' ').trim()
    .replace(/^Error invoking remote method '[^']*': /, '')
    .replace(/^Error: /, '')
    .trim()
    .slice(0, 300)
  return text || 'нет ответа сервера'
}

// Черновик записи живёт в IndexedDB рендерера от остановки до серверного receipt (0013 § 3.3, #109).
const DRAFTS_DB = 'sound-scout-recorder'
const DRAFTS = 'drafts'
const DRAFT_TOKEN = /^[0-9a-f]{12}$/

export function draftStore(idb) {
  let opening = null
  const open = () => opening ??= new Promise((resolve, reject) => {
    const request = idb.open(DRAFTS_DB, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(DRAFTS, { keyPath: 'token' })
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  }).catch(error => { opening = null; throw error })
  const run = async (mode, call) => {
    const db = await open()
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DRAFTS, mode)
      const request = call(tx.objectStore(DRAFTS))
      tx.oncomplete = () => resolve(request.result)
      tx.onerror = tx.onabort = () => reject(tx.error)
    })
  }
  return {
    put: draft => run('readwrite', store => store.put(draft)),
    remove: token => run('readwrite', store => store.delete(token)),
    list: () => run('readonly', store => store.getAll())
  }
}

const isText = value => typeof value === 'string' && value.length > 0
const isOwner = owner => owner === null || (isText(owner?.connectionId) && isText(owner?.profile))

function validDraft(draft) {
  return DRAFT_TOKEN.test(draft?.token) && Number.isFinite(draft.createdAt) &&
    draft.wav?.byteLength > WAV_HEADER_BYTES && isText(draft.target?.stored) && isOwner(draft.target.owner) &&
    isOwner(draft.owner) && typeof draft.submitted === 'boolean' &&
    (draft.attach === null || (isText(draft.attach?.name) && draft.attach.ref_text?.startsWith?.('@file:') === true)) &&
    (draft.attach !== null || !draft.submitted) && (draft.attach === null || draft.owner !== null)
}

const positive = value => Number.isInteger(value) && value > 0
const wholeSeconds = value => Number.isFinite(value) && value > 0 ? Math.floor(value) : 0
const playerClock = seconds => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`

function describeJob(job, result) {
  if (job.state === 'queued') return { text: 'Разбор: в очереди', final: false }
  if (job.state === 'running') {
    if (job.cancel_requested === true) return { text: 'Разбор: отмена запрошена', final: false }
    const stage = job.current_stage
    if (!stage) return { text: 'Разбор: идёт', final: false }
    const done = Array.isArray(job.finished_stages) ? job.finished_stages.length : 0
    const total = Array.isArray(job.stages) ? job.stages.length : 0
    return { text: `Разбор: этап ${done + 1} из ${total} — ${stage.stage}`, final: false }
  }
  if (job.state === 'finished') {
    return { text: `Разбор готов: ${result?.status ?? 'нет результата'}. Итог придёт в чат заметкой`, final: true, kind: 'success' }
  }
  if (job.state === 'failed') return { text: `Разбор не удался: ${job.state_reason}`, final: true, kind: 'error' }
  if (job.state === 'cancelled') return { text: 'Разбор отменён', final: true, kind: 'warning' }
  return null
}

const ENDED = ['finished', 'failed', 'cancelled']
const STAGE_LABELS = { prepare: 'Подготовка', acoustics: 'Измерения', asr: 'Распознавание речи', audio_review: 'Слуховой анализ' }
// mark or stage status → [text, Codicon, tone of the CSS]
const STAGE_MARKS = {
  completed: ['готово', 'pass', 'ok'],
  skipped: ['пропущен', 'circle-slash', 'warn'],
  unavailable: ['недоступен', 'warning', 'warn'],
  failed: ['ошибка', 'error', 'error'],
  current: ['идёт', 'play-circle', 'current'],
  waiting: ['ждёт', 'circle-large-outline', 'idle'],
  missing: ['не выполнен', 'dash', 'idle']
}
const WRONG_CONNECTION = 'Окно подключено к другому серверу. Переключись на подключение этого чата.'
const ACTION_FAILED = { cancel: 'Отменить разбор не удалось', retry: 'Повторить разбор не удалось', report: 'Открыть отчёт не удалось' }

// The planned stages in plan order. A queued job shows no finished_stages: after a retry they are
// those of the previous attempt.
export function stageRows(view) {
  if (!view || !Array.isArray(view.stages)) return []
  const finished = view.state !== 'queued' && Array.isArray(view.finished_stages) ? view.finished_stages : []
  return view.stages.map(stage => {
    const record = finished.find(item => item?.stage === stage)
    const mark = record ? 'done'
      : view.state === 'running' && view.current_stage?.stage === stage ? 'current'
        : ENDED.includes(view.state) ? 'missing' : 'waiting'
    return { stage, label: STAGE_LABELS[stage] ?? String(stage), mark, status: record ? String(record.status) : mark }
  })
}

function attemptNumber(view) {
  const attempts = Number.isInteger(view?.attempts) ? view.attempts : 0
  return view?.state === 'queued' ? attempts + 1 : attempts
}

// `not_retryable` blames the removed recording only when the status says so: the state may
// have changed instead.
function actionError(name, error, view) {
  const code = error?.code || 'no_answer'
  if (code === 'wrong_connection') return WRONG_CONNECTION
  if (name === 'retry' && code === 'not_retryable' && view?.staging_removed_at) return 'Повтор невозможен: исходная запись уже удалена. Запиши заново.'
  if (name === 'retry' && (code === 'queue_full' || code === 'staging_full')) return 'Очередь занята, повтори позже.'
  if (name === 'report' && code === 'no_result') return 'Отчёт недоступен: у разбора пока нет результата.'
  const detail = code === 'no_answer' && error?.message && error.message !== code ? ` — ${error.message}` : ''
  return `${ACTION_FAILED[name]}: ${code}${detail}`
}

function playerError(error) {
  const code = error?.code || 'no_answer'
  if (code === 'recording_missing') return 'Запись не найдена на станции: файла больше нет. Hermes мог его удалить.'
  if (code === 'recording_too_large') return 'Запись больше лимита станции, прослушать её здесь нельзя.'
  if (code === 'wrong_connection') return WRONG_CONNECTION
  if (code === 'no_answer') return `Не удалось получить запись: no_answer — ${error?.message || 'нет ответа сервера'}`
  return `Не удалось получить запись: ${code}`
}

// The WAV bytes of a GET /recordings/{name} answer about that very name, or bad_answer.
function recordingBytes(answer, name) {
  const dataUrl = answer.data_url
  if (answer.name !== name || typeof dataUrl !== 'string' || !dataUrl.startsWith(DATA_URL_PREFIX)) {
    throw new RecorderError('bad_answer')
  }
  let binary
  try { binary = atob(dataUrl.slice(DATA_URL_PREFIX.length)) } catch { throw new RecorderError('bad_answer') }
  if (!binary.length || binary.length !== answer.size_bytes) throw new RecorderError('bad_answer')
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

async function checked(request) {
  let answer
  try {
    answer = await request()
  } catch (error) {
    throw new RecorderError('no_answer', desktopErrorText(error))
  }
  if (answer?.ok !== true) throw new RecorderError(String(answer?.error || 'bad_answer'))
  return answer
}

export class Recorder {
  constructor(host, ctx, {
    sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    now = () => Date.now(),
    pollEveryMs = 3000,
    maxPollFailures = 10,
    random = n => crypto.getRandomValues(new Uint8Array(n)),
    idb = globalThis.indexedDB
  } = {}) {
    this.host = host
    this.ctx = ctx
    this.sleep = sleep
    this.now = now
    this.pollEveryMs = pollEveryMs
    this.maxPollFailures = maxPollFailures
    this.random = random
    this.state = atom({
      phase: 'idle', error: '', startedAt: 0, limitSeconds: 0, canAnalyze: false, intakeError: '', restored: false, unprotected: false
    })
    this.elapsed = atom(0)
    this.level = atom(0)
    this.job = atom(null)
    // stored chat → { name, owner } of the last recording sent there, in the memory of this window
    this.sent = atom({})
    this.player = atom(null)
    this.playerGeneration = 0
    // The one media element of this window, outside React: a render never creates or resets it.
    this.audio = null
    this.audioListeners = []
    // The dialog shows only in the composer of the player's chat: a chat switch closes the player.
    const focused = host.state?.focusedStoredSessionId
    this.unwatchFocus = typeof focused?.listen === 'function' ? focused.listen(() => this.focusChanged()) : null
    this.levelAt = -Infinity
    this.chunks = []
    this.generation = 0
    this.pollGeneration = 0
    this.polling = null
    this.pending = null
    this.target = null
    this.budget = null
    this.drafts = draftStore(idb)
    this.draftWrites = Promise.resolve()
    this.storedTokens = new Set()
    this.droppedTokens = new Set()
  }

  // Записи в хранилище идут цепочкой по одной, запись и отправка их не ждут.
  // Зависшая операция IndexedDB останавливает все следующие записи хранилища до перезагрузки окна.
  draftWrite(step) {
    this.draftWrites = this.draftWrites.then(step).catch(() => {})
  }

  saveDraft(pending, target) {
    const snapshot = {
      token: pending.token, createdAt: pending.createdAt, target, owner: pending.owner, attach: pending.attach, submitted: pending.submitted
    }
    const mark = unprotected => { if (this.pending === pending) this.update({ unprotected }) }
    this.draftWrite(async () => {
      try {
        pending.wav ??= await pending.blob.arrayBuffer()
        await this.drafts.put({ ...snapshot, wav: pending.wav })
        this.storedTokens.add(pending.token)
      } catch {
        mark(true)
        return
      }
      mark(false)
    })
  }

  dropDraft(token) {
    this.droppedTokens.add(token)
    this.draftWrite(async () => {
      const stored = this.storedTokens.delete(token)
      try {
        await this.drafts.remove(token)
      } catch {
        if (stored) {
          this.host.notify({
            kind: 'warning', message: 'Sound Scout: сохранённую запись не удалось удалить из хранилища окна. Она снова появится при следующей загрузке.'
          })
        }
      }
    })
  }

  // Запись предлагается сразу после чтения хранилища; ответ /health только открывает разбор.
  async restore() {
    let drafts
    try { drafts = await this.drafts.list() } catch { return }
    for (const draft of drafts) if (!validDraft(draft)) this.dropDraft(draft?.token)
    const draft = drafts
      .filter(draft => validDraft(draft) && !this.droppedTokens.has(draft.token))
      .sort((a, b) => a.createdAt - b.createdAt)[0]
    if (!draft || this.state.get().phase !== 'idle') return
    this.storedTokens.add(draft.token)
    this.target = { stored: draft.target.stored, owner: draft.target.owner }
    const pending = this.pending = {
      blob: new Blob([draft.wav], { type: 'audio/wav' }), wav: draft.wav, dataUrl: null, token: draft.token,
      createdAt: draft.createdAt, owner: draft.owner, attach: draft.attach, submitted: draft.submitted, job: null
    }
    this.update({ phase: 'ready', restored: true, unprotected: false, error: '', canAnalyze: false, intakeError: 'health_pending' })
    this.host.notify({ kind: 'info', message: 'Sound Scout: есть неотправленная запись. Отправь её в исходный чат или удали.' })
    const health = await this.readBudget().catch(error => ({ canAnalyze: false, intakeError: error.code }))
    if (this.pending === pending) this.update({ canAnalyze: health.canAnalyze, intakeError: health.intakeError })
  }

  update(values) {
    this.state.set({ ...this.state.get(), ...values })
    if (values.error) this.host.notify({ kind: 'error', message: values.error })
  }

  async readBudget() {
    let answer
    try {
      answer = await this.ctx.rest('/health', { timeoutMs: HEALTH_TIMEOUT_MS })
    } catch (error) {
      throw new RecorderError('no_answer', desktopErrorText(error))
    }
    if (answer?.ok !== true) throw new RecorderError(String(answer?.error || 'bad_answer'))
    const budgets = answer.budgets
    if (!positive(budgets?.max_duration_seconds) || !positive(budgets?.max_input_bytes)) {
      throw new RecorderError(String(answer.intake?.error || 'no_budgets'))
    }
    return {
      budget: { max_duration_seconds: budgets.max_duration_seconds, max_input_bytes: budgets.max_input_bytes },
      canAnalyze: answer.intake?.ready === true,
      intakeError: answer.intake?.ready === true ? '' : String(answer.intake?.error || 'intake_unknown')
    }
  }

  async start() {
    if (this.state.get().phase !== 'idle') return
    this.level.set(0)
    this.levelAt = -Infinity
    this.elapsed.set(0)
    const generation = ++this.generation
    const owner = this.host.state.focusedSessionOwner.get()
    const target = {
      stored: this.host.state.focusedStoredSessionId.get(),
      owner: owner ? { connectionId: owner.connectionId, profile: owner.profile } : null
    }
    this.pending = null
    this.update({ phase: 'starting', error: '', restored: false, unprotected: false })
    if (!target.stored) {
      this.update({ phase: 'idle', error: 'Открой существующий чат перед записью.' })
      return
    }
    let health
    try {
      health = await this.readBudget()
    } catch (error) {
      if (generation !== this.generation) return
      health = { budget: R02_BUDGET, canAnalyze: false, intakeError: error.code }
      this.host.notify({
        kind: 'warning',
        message: `Sound Scout не сообщил лимит записи: ${error.message}. Запись идёт с лимитом 180 с, разбор недоступен.`
      })
    }
    if (generation !== this.generation) return
    this.target = target
    this.budget = health.budget
    this.update({ limitSeconds: health.budget.max_duration_seconds, canAnalyze: health.canAnalyze, intakeError: health.intakeError })
    let message = 'Не удалось получить доступ к микрофону. Повтори запись.'
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: { ideal: 2 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false
      } })
      if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return }
      this.stream = stream
      const settings = stream.getAudioTracks?.()[0]?.getSettings?.() || {}
      message = 'Микрофон не отключил автоматическую обработку звука. Проверь настройки устройства.'
      if (['echoCancellation', 'noiseSuppression', 'autoGainControl'].some(key => settings[key] === true)) {
        throw new Error('Микрофон не отключил автоматическую обработку звука')
      }
      message = 'Не удалось настроить запись с микрофона. Повтори запись.'
      this.context = new AudioContext()
      await this.context.resume()
      if (generation !== this.generation) return
      this.sampleRate = this.context.sampleRate
      this.source = this.context.createMediaStreamSource(this.stream)
      // ScriptProcessor works without a worklet asset, as in Voice Note.
      this.processor = this.context.createScriptProcessor(4096, 2, 1)
      this.chunks = []
      this.wavBytes = WAV_HEADER_BYTES
      this.frames = 0
      const maxBytes = this.budget.max_input_bytes
      // Длину WAV ограничивает счёт кадров; таймер ниже — только страховка.
      const maxFrames = this.budget.max_duration_seconds * this.sampleRate
      this.processor.onaudioprocess = event => {
        if (generation !== this.generation || this.state.get().phase !== 'recording') return
        const left = maxFrames - this.frames
        let chunk = Array.from({ length: event.inputBuffer.numberOfChannels },
          (_, channel) => new Float32Array(event.inputBuffer.getChannelData(channel)))
        const full = chunk[0].length >= left
        if (full) chunk = chunk.map(channel => channel.subarray(0, left))
        const next = this.wavBytes + chunk[0].length * chunk.length * 2
        if (next > maxBytes) { this.reachLimit(); return }
        if (chunk[0].length) {
          this.wavBytes = next
          this.frames += chunk[0].length
          this.chunks.push(chunk)
        }
        if (full) { this.reachLimit(); return }
        const now = this.now()
        if (now - this.levelAt < 100) return
        this.levelAt = now
        let level = 0
        for (const channel of chunk) {
          let sum = 0
          for (const sample of channel) sum += sample * sample
          const rms = Math.sqrt(sum / channel.length)
          if (Number.isFinite(rms)) level = Math.max(level, rms)
        }
        this.level.set(Math.min(1, level))
      }
      this.source.connect(this.processor)
      this.processor.connect(this.context.destination)
      const startedAt = this.now()
      this.update({ phase: 'recording', startedAt })
      this.limitTimer = setTimeout(() => this.reachLimit(), this.budget.max_duration_seconds * 1000)
      this.limitTimer?.unref?.()
      this.ticker = setInterval(() => this.elapsed.set(Math.max(0, Math.floor((this.now() - startedAt) / 1000))), 1000)
      this.ticker?.unref?.()
    } catch (error) {
      if (generation !== this.generation) return
      if (!this.stream && error?.name === 'NotAllowedError') message = 'Нет доступа к микрофону. Разреши его использование в настройках системы и Hermes.'
      if (!this.stream && error?.name === 'NotFoundError') message = 'Микрофон не найден. Подключи или выбери устройство ввода.'
      if (!this.stream && error?.name === 'NotReadableError') message = 'Микрофон занят или недоступен. Проверь устройство и другие приложения.'
      this.releaseCapture()
      this.update({ phase: 'idle', error: message })
      throw error
    }
  }

  reachLimit() {
    if (this.state.get().phase !== 'recording') return
    try { this.stop() } catch { return }
    this.update({ error: `Достигнут лимит записи ${this.budget.max_duration_seconds} с. Можно отправить записанное.` })
  }

  releaseCapture() {
    this.level.set(0)
    this.levelAt = -Infinity
    clearTimeout(this.limitTimer)
    clearInterval(this.ticker)
    this.limitTimer = this.ticker = null
    if (this.processor) { this.processor.onaudioprocess = null; this.processor.disconnect() }
    this.source?.disconnect()
    this.stream?.getTracks().forEach(track => track.stop())
    this.context?.close().catch(() => {})
    this.processor = this.source = this.stream = this.context = null
  }

  cancel() {
    const token = this.pending?.token
    this.discard()
    if (token) this.dropDraft(token)
  }

  // Выгрузка плагина не удаляет черновик: его предложит следующая загрузка.
  discard() {
    this.generation++
    this.releaseCapture()
    this.chunks = []
    this.pending = null
    this.update({ phase: 'idle', error: '', restored: false })
  }

  dispose() {
    this.pollGeneration++
    this.unwatchFocus?.()
    this.unwatchFocus = null
    this.closePlayer()
    this.discard()
  }

  stop() {
    if (this.state.get().phase !== 'recording') return
    let blob
    try { blob = encodeWav(this.chunks, this.sampleRate) }
    catch (error) {
      this.update({ phase: 'idle', error: 'В записи пока нет звука. Попробуй записать ещё раз.' })
      throw error
    }
    finally { this.chunks = []; this.releaseCapture() }
    const token = Array.from(this.random(6), byte => byte.toString(16).padStart(2, '0')).join('')
    this.pending = { blob, dataUrl: null, token, createdAt: this.now(), owner: null, attach: null, submitted: false, job: null }
    this.update({ phase: 'ready' })
    this.saveDraft(this.pending, this.target)
  }

  async resolveRoute(target) {
    let owner = target.owner
    if (!owner && this.host.state.focusedStoredSessionId.get() === target.stored) {
      const current = this.host.state.focusedSessionOwner.get()
      if (current) owner = { connectionId: current.connectionId, profile: current.profile }
    }
    return this.findRoute(owner)
  }

  async findRoute(owner) {
    const routes = await this.host.profileRoutes()
    const matches = owner
      ? routes.filter(route => route.connectionId === owner.connectionId && route.profile === owner.profile)
      : routes
    if (matches.length !== 1) throw new RecorderError('ambiguous_route')
    return matches[0]
  }

  async send({ analyze = false } = {}) {
    const phase = this.state.get().phase
    if (phase === 'recording') this.stop()
    else if (phase !== 'ready') return
    const pending = this.pending
    const target = this.target
    const generation = this.generation
    if (!pending) return
    this.update({ phase: 'sending', error: '' })
    let message = 'Не удалось подключить исходный чат. Повтори отправку после восстановления связи.'
    try {
      if (!pending.attach || !pending.submitted) {
        // Повтор после attach идёт по маршруту attach, а не по текущему фокусу.
        const route = await (pending.attach ? this.findRoute(pending.owner) : this.resolveRoute(target)).catch(error => {
          if (error?.code === 'ambiguous_route') message = 'Не удалось однозначно определить сервер исходного чата.'
          throw error
        })
        if (generation !== this.generation) return
        pending.owner = { connectionId: route.connectionId, profile: route.profile, targetProfile: route.targetProfile }
        let release
        try {
          release = await this.host.retainProfile(route, FOREGROUND)
          if (generation !== this.generation) return
          message = 'Не удалось открыть исходный чат на сервере. Повтори отправку.'
          const resumed = await this.host.requestProfile(route, 'session.resume',
            { session_id: target.stored, omit_messages: true }, undefined, FOREGROUND)
          if (generation !== this.generation) return
          if (typeof resumed?.session_id !== 'string' || !resumed.session_id) throw new Error('Чат недоступен')
          if (!pending.attach) {
            message = 'Не удалось подготовить аудио к отправке. Повтори отправку.'
            pending.dataUrl ??= await readDataUrl(pending.blob)
            if (generation !== this.generation) return
            message = 'Не удалось загрузить аудио на сервер. Повтори отправку.'
            const name = `sound-scout-${pending.token}.wav`
            const uploaded = await this.host.requestProfile(route, 'file.attach',
              { session_id: resumed.session_id, name, data_url: pending.dataUrl }, ATTACH_TIMEOUT_MS)
            if (generation !== this.generation) return
            message = 'Сервер не подтвердил получение аудио. Повтори отправку.'
            if (uploaded?.attached !== true || typeof uploaded.ref_text !== 'string' || !uploaded.ref_text.startsWith('@file:')) {
              throw new Error('Сервер не принял аудио')
            }
            pending.attach = { ref_text: uploaded.ref_text, name }
            this.saveDraft(pending, target)
          }
          message = 'Чат не принял сообщение. Аудио уже загружено, повтори отправку.'
          const receipt = await this.host.requestProfile(route, 'prompt.submit', {
            session_id: resumed.session_id, text: pending.attach.ref_text, submission_id: `sound-scout-${pending.token}`
          })
          if (generation !== this.generation) return
          if (typeof receipt?.admission_id !== 'string' || !receipt.admission_id) throw new Error('Чат не подтвердил сообщение')
          pending.submitted = true
          if (analyze) this.saveDraft(pending, target)
        } finally { release?.() }
      }
      if (RECORDING_NAME.test(pending.attach.name)) {
        this.sent.set({ ...this.sent.get(), [target.stored]: { name: pending.attach.name, owner: pending.owner } })
      }
      if (analyze && !pending.job) {
        message = null
        await this.submitJob(pending, target, generation)
        if (generation !== this.generation) return
      }
      this.pending = null
      this.dropDraft(pending.token)
      this.update({ phase: 'idle', restored: false })
      this.host.notify({ kind: 'success', message: analyze ? 'Голосовое отправлено, разбор принят' : 'Голосовое отправлено' })
    } catch (error) {
      if (generation !== this.generation) return
      const code = error?.code || 'no_answer'
      const detail = code === 'no_answer' && error?.message && error.message !== code ? ` — ${error.message}` : ''
      const text = message === null
        ? `Голосовое в чате, но разбор не принят: ${code}${detail}. Повтори «Отправить и разобрать».`
        : `${message} Запись сохранена.`
      this.update({ phase: 'ready', error: text })
      throw error
    }
  }

  async submitJob(pending, target, generation) {
    const owner = pending.owner
    if (!this.state.get().canAnalyze) throw new RecorderError('intake_not_ready')
    if (typeof owner.targetProfile !== 'string' || !owner.targetProfile) throw new RecorderError('no_target_profile')
    pending.dataUrl ??= await readDataUrl(pending.blob)
    if (generation !== this.generation) return
    // ctx.rest идёт в backend окна, поэтому окно должно стоять на маршруте чата.
    if (!this.onRoute(owner)) throw new RecorderError('wrong_connection')
    const query = new URLSearchParams({
      request_id: `sound-scout-desktop-${pending.token}`, session_id: target.stored, profile: owner.targetProfile
    })
    let answer
    try {
      answer = await this.ctx.rest(`/jobs?${query}`, {
        method: 'POST',
        body: { data_url: pending.dataUrl },
        timeoutMs: UPLOAD_TIMEOUT_MS
      })
    } catch (error) {
      throw new RecorderError('no_answer', desktopErrorText(error))
    }
    if (generation !== this.generation) return
    if (answer?.ok !== true) throw new RecorderError(String(answer?.error || 'bad_answer'))
    if (typeof answer.job_id !== 'string' || !JOB_ID.test(answer.job_id)) throw new RecorderError('bad_answer')
    pending.job = { job_id: answer.job_id }
    this.track(answer.job_id, owner, target.stored, pending.attach.name)
  }

  onRoute(owner) {
    const state = this.host.state
    return state.connectionId?.get?.() === owner.connectionId && state.profile?.get?.() === owner.profile
  }

  track(jobId, owner, sessionId, recording) {
    this.job.set({
      jobId, sessionId, owner, recording, view: null, result: null, text: 'Разбор: принят',
      final: false, stopped: false, busy: null, error: '', report: null
    })
    this.follow(false)
  }

  setJob(values) {
    const job = this.job.get()
    if (job) this.job.set({ ...job, ...values })
  }

  follow(immediate) {
    const { jobId, owner } = this.job.get()
    const poll = ++this.pollGeneration
    this.polling = this.poll(jobId, owner, poll, immediate)
    return this.polling
  }

  // One request at once, then the cycle again unless the job ended.
  refresh() {
    if (!this.job.get()) return
    this.setJob({ stopped: false })
    return this.follow(true)
  }

  async poll(jobId, owner, poll, immediate = false) {
    let failures = 0
    let wait = !immediate
    while (poll === this.pollGeneration) {
      if (wait) await this.sleep(this.pollEveryMs)
      wait = true
      if (poll !== this.pollGeneration) return
      if (!this.onRoute(owner)) continue
      let answer
      let failure = null
      try {
        answer = await this.ctx.rest(`/jobs/${jobId}`, { timeoutMs: STATUS_TIMEOUT_MS })
      } catch (error) {
        failure = new RecorderError('no_answer', desktopErrorText(error))
      }
      if (poll !== this.pollGeneration) return
      const view = answer?.ok === true && answer.job?.job_id === jobId ? answer.job : null
      const seen = view ? describeJob(view, answer.result) : null
      if (seen) {
        failures = 0
        this.setJob({ view, result: answer.result ?? null, text: seen.text, final: seen.final, stopped: false })
        if (seen.final) {
          this.host.notify({ kind: seen.kind, message: seen.text })
          return
        }
        continue
      }
      if (++failures >= this.maxPollFailures) {
        const text = failure
          ? `Статус разбора недоступен: ${failure.code} — ${failure.message}`
          : `Статус разбора недоступен: ${String(answer?.ok === false && answer.error ? answer.error : 'bad_answer')}`
        this.setJob({ text, stopped: true })
        this.host.notify({ kind: 'warning', message: text })
        return
      }
    }
  }

  // Each action names the job of the card; a second click while one is in flight is ignored.
  async act(name, run) {
    const job = this.job.get()
    if (!job || job.busy) return
    if (!this.onRoute(job.owner)) {
      this.setJob({ error: WRONG_CONNECTION })
      return
    }
    const current = () => this.job.get()?.jobId === job.jobId
    this.setJob({ busy: name, error: '' })
    try {
      await run(job.jobId, current)
    } catch (error) {
      if (!current()) return
      const report = this.job.get().report
      if (name === 'report') {
        if (report?.open) this.setJob({ report: { ...report, loading: false, error: actionError(name, error) } })
      } else {
        this.setJob({ error: actionError(name, error, this.job.get().view) })
      }
    } finally {
      if (current()) this.setJob({ busy: null })
    }
  }

  adopt(view) {
    const job = this.job.get()
    if (view?.job_id !== job?.jobId) return
    const seen = describeJob(view, job.result)
    if (seen) this.setJob({ view, text: seen.text, final: seen.final })
  }

  // The job is cancelled first; the turn of the chat of the job stops only after that (#188),
  // also when a newer job took the card meanwhile — unless that job went into the same chat,
  // whose running turn can then be the one the newer job started.
  cancelJob() {
    return this.act('cancel', async (jobId, current) => {
      const { owner, sessionId } = this.job.get()
      const answer = await checked(() => this.ctx.rest(`/jobs/${jobId}/cancel`, { method: 'POST', body: {}, timeoutMs: STATUS_TIMEOUT_MS }))
      if (current()) {
        this.adopt(answer.job)
        this.refresh()
      } else if (this.job.get()?.sessionId === sessionId) return
      try {
        await this.stopTurn(owner, sessionId)
      } catch (error) {
        if (current()) this.setJob({ error: `Разбор отменён, но ход чата не остановлен: ${desktopErrorText(error)}` })
      }
    })
  }

  // The Stop of Desktop: session.interrupt of the runtime session of this chat (decision 0029).
  // A chat without a running turn gets no interrupt.
  async stopTurn(owner, stored) {
    const route = await this.findRoute(owner)
    const release = await this.host.retainProfile(route, FOREGROUND)
    try {
      const resumed = await this.host.requestProfile(route, 'session.resume',
        { session_id: stored, omit_messages: true }, undefined, FOREGROUND)
      if (typeof resumed?.session_id !== 'string' || !resumed.session_id) throw new Error('Чат недоступен')
      if (resumed.running !== true) return
      const params = { session_id: resumed.session_id }
      if (Number.isInteger(resumed.execution_generation)) params.execution_generation = resumed.execution_generation
      await this.host.requestProfile(route, 'session.interrupt', params, undefined, FOREGROUND)
    } finally { release() }
  }

  // One status request after a refused or lost retry: the card shows the state the job has now,
  // or Refresh when that request fails too.
  async reconcile(jobId, current) {
    let answer = null
    try {
      answer = await this.ctx.rest(`/jobs/${jobId}`, { timeoutMs: STATUS_TIMEOUT_MS })
    } catch {}
    if (!current()) return
    const view = answer?.ok === true && answer.job?.job_id === jobId ? answer.job : null
    const seen = view ? describeJob(view, answer.result) : null
    if (!seen) {
      this.setJob({ stopped: true })
      return
    }
    this.setJob({ view, result: answer.result ?? null, text: seen.text, final: seen.final, stopped: false })
    if (!seen.final) this.follow(false)
  }

  retryJob() {
    return this.act('retry', async (jobId, current) => {
      let answer
      try {
        answer = await checked(() => this.ctx.rest(`/jobs/${jobId}/retry`, { method: 'POST', body: {}, timeoutMs: STATUS_TIMEOUT_MS }))
      } catch (error) {
        if (current()) await this.reconcile(jobId, current)
        throw error
      }
      if (!current()) return
      this.setJob({ final: false, stopped: false, report: null, result: null })
      this.adopt(answer.job)
      this.refresh()
    })
  }

  // No cache: after a retry the next click reads the report of the new attempt.
  openReport() {
    return this.act('report', async (jobId, current) => {
      this.setJob({ report: { open: true, loading: true, markdown: '', truncated: false, attempt: null, error: '' } })
      const answer = await checked(() => this.ctx.rest(`/jobs/${jobId}/report`, { timeoutMs: STATUS_TIMEOUT_MS }))
      const report = answer.report
      if (report?.job_id !== jobId || typeof report.markdown !== 'string') throw new RecorderError('bad_answer')
      if (!current() || !this.job.get().report?.open) return
      this.setJob({ report: {
        open: true, loading: false, markdown: report.markdown, truncated: report.truncated === true,
        attempt: Number.isInteger(report.attempt) ? report.attempt : null, error: '',
        runDir: typeof report.run_dir === 'string' ? report.run_dir : null
      } })
    })
  }

  closeReport() {
    this.setJob({ report: null })
  }

  dismiss() {
    const job = this.job.get()
    if (!job || !(job.final || job.stopped)) return
    this.pollGeneration++
    this.job.set(null)
  }

  // Each open reads the file of the station again: no cache, the recorder buffer is never played.
  async openPlayer({ sessionId, name, owner, origin }) {
    if (this.player.get()?.loading) return
    if (this.player.get()) this.closePlayer()
    const jobId = origin === 'card' ? this.job.get()?.jobId : null
    const show = error => {
      if (origin === 'recorder') this.update({ error })
      else if (this.job.get()?.jobId === jobId) this.setJob({ error })
    }
    if (!this.onRoute(owner)) {
      show(WRONG_CONNECTION)
      return
    }
    show('')
    const generation = ++this.playerGeneration
    this.player.set({ sessionId, name, owner, origin, loading: true, url: null, error: '' })
    let bytes
    try {
      const answer = await checked(() => this.ctx.rest(`/recordings/${name}`, { timeoutMs: RECORDING_TIMEOUT_MS }))
      bytes = recordingBytes(answer, name)
    } catch (error) {
      if (generation !== this.playerGeneration) return
      this.player.set(null)
      show(playerError(error))
      return
    }
    const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
    if (generation !== this.playerGeneration) {
      URL.revokeObjectURL(url)
      return
    }
    const audio = new Audio()
    audio.preload = 'auto'
    this.audio = audio
    this.audioListeners = [
      ...PLAYER_EVENTS.map(type => [type, () => this.syncPlayer(audio)]),
      ['error', () => this.playerFailed(audio)]
    ]
    for (const [type, listener] of this.audioListeners) audio.addEventListener(type, listener)
    audio.src = url
    this.player.set({ ...this.player.get(), loading: false, url, playing: false, position: 0, duration: 0 })
  }

  focusChanged() {
    const player = this.player.get()
    if (player && this.host.state.focusedStoredSessionId.get() !== player.sessionId) this.closePlayer()
  }

  // The atom changes only when a whole second or the play state changes.
  syncPlayer(audio) {
    const player = this.player.get()
    if (audio !== this.audio || !player?.url) return
    const playing = !audio.paused && !audio.ended
    const position = wholeSeconds(audio.currentTime)
    const duration = wholeSeconds(audio.duration)
    if (player.playing === playing && player.position === position && player.duration === duration) return
    this.player.set({ ...player, playing, position, duration })
  }

  playerFailed(audio) {
    const player = this.player.get()
    if (audio !== this.audio || !player?.url) return
    const code = MEDIA_ERRORS[audio.error?.code] ?? 'MEDIA_ERR_UNKNOWN'
    this.player.set({ ...player, playing: false, error: `Плеер не может прочитать запись: ${code}` })
  }

  async togglePlay() {
    const audio = this.audio
    if (!audio) return
    if (!audio.paused && !audio.ended) {
      audio.pause()
      this.syncPlayer(audio)
      return
    }
    const player = this.player.get()
    if (player?.error) this.player.set({ ...player, error: '' })
    try {
      await audio.play()
    } catch (error) {
      const current = this.player.get()
      if (audio === this.audio && current) {
        this.player.set({ ...current, playing: false, error: `Не удалось начать воспроизведение: ${error?.name || 'Error'}` })
      }
      return
    }
    this.syncPlayer(audio)
  }

  seek(delta) {
    const audio = this.audio
    if (!audio) return
    const end = Number.isFinite(audio.duration) ? audio.duration : Infinity
    audio.currentTime = Math.min(Math.max(audio.currentTime + delta, 0), end)
    this.syncPlayer(audio)
  }

  restart() {
    const audio = this.audio
    if (!audio) return
    audio.currentTime = 0
    this.syncPlayer(audio)
  }

  closePlayer() {
    this.playerGeneration++
    const audio = this.audio
    this.audio = null
    if (audio) {
      for (const [type, listener] of this.audioListeners) audio.removeEventListener(type, listener)
      audio.pause()
      audio.removeAttribute('src')
      audio.load()
    }
    this.audioListeners = []
    const url = this.player.get()?.url
    this.player.set(null)
    if (!url) return
    const timer = setTimeout(() => URL.revokeObjectURL(url), URL_RELEASE_MS)
    timer?.unref?.()
  }

  // The same Blob the player plays, saved by the download of the renderer under the recording name.
  downloadRecording() {
    const player = this.player.get()
    if (!player?.url) return
    let link
    try {
      link = document.createElement('a')
      link.href = player.url
      link.download = player.name
      link.rel = 'noopener noreferrer'
      link.style.display = 'none'
      document.body.appendChild(link)
      link.click()
    } catch (error) {
      if (this.player.get() === player) this.player.set({ ...player, error: `Не удалось сохранить запись: ${desktopErrorText(error)}` })
    } finally {
      link?.remove()
    }
  }
}

function SignalMeter({ recorder }) {
  const rms = useValue(recorder.level)
  const level = Math.max(0, Math.min(1, (20 * Math.log10(Math.max(rms, 1e-6)) + 60) / 48))
  return jsx('span', {
    className: 'sound-scout-recorder__meter', role: 'meter', 'aria-label': 'Входной сигнал',
    'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(level * 100), title: 'Сигнал с микрофона',
    children: [6, 12, 8].map((height, index) => jsx('span', {
      'aria-hidden': true, style: { height, transform: `scaleY(${0.2 + level * 0.8})`, opacity: 0.3 + level * 0.7 }
    }, index))
  })
}

function action({ label, icon, title, disabled, onClick }) {
  return jsx(Button, {
    type: 'button', variant: 'ghost', size: 'sm', title, 'aria-label': label, disabled,
    onClick: () => Promise.resolve().then(onClick).catch(() => {}),
    children: jsxs('span', { className: 'sound-scout-recorder__label', children: [
      jsx(Codicon, { name: icon, size: 14, 'aria-hidden': true }), jsx('span', { children: label })
    ] })
  }, label)
}

const STYLE = `
  .sound-scout-recorder { position:relative; display:inline-flex; align-items:center; gap:4px; max-width:100%; min-height:28px; }
  .sound-scout-recorder__status { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; color:var(--ui-text-secondary); font-variant-numeric:tabular-nums; }
  .sound-scout-recorder__dot { width:6px; height:6px; border-radius:50%; background:var(--ui-red); }
  .sound-scout-recorder__meter { display:inline-flex; align-items:center; gap:2px; width:14px; height:12px; color:var(--ui-text-secondary); }
  .sound-scout-recorder__meter span { display:block; width:2px; border-radius:1px; background:var(--ui-text-secondary); }
  .sound-scout-recorder__label { display:inline-flex; align-items:center; gap:5px; }
  .sound-scout-player { display:flex; flex-direction:column; gap:8px; color:var(--ui-text-primary); }
  .sound-scout-player__time { color:var(--ui-text-secondary); font-variant-numeric:tabular-nums; }
  .sound-scout-player__actions { display:flex; flex-wrap:wrap; gap:4px; }
  .sound-scout-player__error { color:var(--ui-red); }
  .sound-scout-recorder__error { position:absolute; bottom:calc(100% + 6px); left:0; z-index:10; width:max-content; max-width:min(260px,80vw); padding:6px 8px; border:1px solid var(--ui-stroke-secondary); border-radius:4px; background:var(--ui-bg-quaternary); color:var(--ui-text-secondary); font-size:12px; }
  @media (prefers-reduced-motion: reduce) { .sound-scout-recorder *, .sound-scout-player, .sound-scout-player * { transition: none !important; animation: none !important; } }
`

const CARD_STYLE = `
  .sound-scout-card { display:flex; flex-direction:column; gap:6px; margin:0 0 6px; padding:8px 10px; border:1px solid var(--ui-stroke-secondary); border-radius:6px; background:var(--ui-bg-elevated); color:var(--ui-text-primary); font-size:12px; }
  .sound-scout-card__head { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 12px; }
  .sound-scout-card__meta { color:var(--ui-text-secondary); }
  .sound-scout-card__stages { display:flex; flex-wrap:wrap; gap:4px 12px; margin:0; padding:0; list-style:none; }
  .sound-scout-card__stage { display:inline-flex; align-items:center; gap:4px; color:var(--ui-text-secondary); }
  .sound-scout-card__stage[data-tone="ok"] { color:var(--ui-green); }
  .sound-scout-card__stage[data-tone="warn"] { color:var(--ui-yellow); }
  .sound-scout-card__stage[data-tone="error"] { color:var(--ui-red); }
  .sound-scout-card__stage[data-tone="current"] { color:var(--ui-text-primary); }
  .sound-scout-card__error { color:var(--ui-red); }
  .sound-scout-card__actions { display:flex; flex-wrap:wrap; gap:4px; }
  .sound-scout-card__report { max-height:70vh; overflow:auto; margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; color:var(--ui-text-primary); }
  .sound-scout-card__note { color:var(--ui-text-secondary); }
  @media (prefers-reduced-motion: reduce) { .sound-scout-card *, .sound-scout-card__dialog, .sound-scout-card__dialog * { transition: none !important; animation: none !important; } }
`

function ReportDialog({ recorder, job }) {
  const report = job.report
  const description = report.attempt ? `Попытка ${report.attempt} · ${job.result?.status ?? 'нет результата'}` : 'Загрузка…'
  const body = report.loading ? [jsx('p', { className: 'sound-scout-card__note', children: 'Загрузка отчёта…' }, 'loading')]
    : report.error ? [jsx('p', { role: 'alert', className: 'sound-scout-card__error', children: report.error }, 'error')]
      : [
          jsx('pre', { className: 'sound-scout-card__report', tabIndex: 0, 'aria-label': 'Текст отчёта', children: report.markdown }, 'text'),
          report.truncated && jsx('p', {
            className: 'sound-scout-card__note',
            children: report.runDir
              ? `Показаны первые 256 KiB. Полный отчёт: \`${report.runDir}/report.md\` в data root на станции`
              : 'Показаны первые 256 KiB. Полный отчёт: report.md в run-каталоге на станции'
          }, 'truncated')
        ]
  return jsx(Dialog, {
    open: true,
    onOpenChange: open => { if (!open) recorder.closeReport() },
    children: jsxs(DialogContent, { className: 'sound-scout-card__dialog', children: [
      jsxs(DialogHeader, { children: [
        jsx(DialogTitle, { children: 'Отчёт Sound Scout' }),
        jsx(DialogDescription, { children: description })
      ] }, 'header'),
      ...body
    ] })
  }, 'report')
}

// Reads the player atom only: the element lives in recorder.audio, so a poll or a second composer of
// the chat re-renders the buttons and never the playback.
function PlayerDialog({ recorder, player }) {
  const body = [
    jsx('p', {
      role: 'status', className: 'sound-scout-player__time', 'aria-label': 'Позиция записи',
      children: `${playerClock(player.position ?? 0)} / ${playerClock(player.duration ?? 0)}`
    }, 'time'),
    jsx('div', { className: 'sound-scout-player__actions', children: [
      player.playing
        ? action({ label: 'Пауза', icon: 'debug-pause', title: 'Приостановить запись', onClick: () => recorder.togglePlay() })
        : action({ label: 'Пуск', icon: 'play', title: 'Воспроизвести запись', onClick: () => recorder.togglePlay() }),
      action({ label: '−5 с', icon: 'arrow-left', title: 'Назад на 5 секунд', onClick: () => recorder.seek(-SEEK_STEP_S) }),
      action({ label: '+5 с', icon: 'arrow-right', title: 'Вперёд на 5 секунд', onClick: () => recorder.seek(SEEK_STEP_S) }),
      action({ label: 'В начало', icon: 'debug-step-back', title: 'Перемотать запись в начало', onClick: () => recorder.restart() }),
      action({ label: 'Скачать', icon: 'cloud-download', title: 'Сохранить этот WAV на компьютер', onClick: () => recorder.downloadRecording() })
    ] }, 'actions')
  ]
  if (player.error) body.push(jsx('p', { role: 'alert', className: 'sound-scout-player__error', children: player.error }, 'error'))
  return jsx(Dialog, {
    open: true,
    onOpenChange: open => { if (!open) recorder.closePlayer() },
    children: jsxs(DialogContent, { className: 'sound-scout-player', children: [
      jsxs(DialogHeader, { children: [
        jsx(DialogTitle, { children: 'Запись Sound Scout' }),
        jsx(DialogDescription, { children: `Файл ${player.name} со станции` })
      ] }, 'header'),
      ...body
    ] })
  }, 'player')
}

export function JobCard({ recorder }) {
  const job = useValue(recorder.job)
  const player = useValue(recorder.player)
  const composer = useComposerSession()
  if (!job || composer?.storedSessionId !== job.sessionId) return null
  const view = job.view
  const state = view?.state
  const attempt = attemptNumber(view)
  const disabled = job.busy !== null
  const head = [jsx('span', { role: 'status', 'aria-live': 'polite', children: job.text }, 'status')]
  if (attempt > 1) head.push(jsx('span', { className: 'sound-scout-card__meta', children: `Попытка ${attempt}` }, 'attempt'))
  if (job.result?.status && state !== 'finished') {
    head.push(jsx('span', { className: 'sound-scout-card__meta', children: `Результат: ${job.result.status}` }, 'result'))
  }
  if (player?.loading && player.origin === 'card') {
    head.push(jsx('span', { className: 'sound-scout-card__meta', children: 'Загрузка записи…' }, 'player'))
  }
  const children = [
    jsx('style', { children: CARD_STYLE }, 'style'),
    jsxs('div', { className: 'sound-scout-card__head', children: head }, 'head')
  ]
  const rows = stageRows(view)
  if (rows.length) {
    children.push(jsx('ol', { className: 'sound-scout-card__stages', 'aria-label': 'Этапы разбора', children: rows.map(row => {
      const [text, icon, tone] = STAGE_MARKS[row.mark === 'done' ? row.status : row.mark] ?? [row.status, 'circle-filled', 'idle']
      return jsxs('li', { className: 'sound-scout-card__stage', 'data-tone': tone, children: [
        jsx(Codicon, { name: icon, size: 14, 'aria-hidden': true }),
        jsx('span', { children: `${row.label}: ${text}` })
      ] }, row.stage)
    }) }, 'stages'))
  }
  if (job.error) children.push(jsx('p', { role: 'alert', className: 'sound-scout-card__error', children: job.error }, 'error'))
  const buttons = []
  if (RECORDING_NAME.test(job.recording ?? '')) {
    buttons.push(action({
      label: 'Прослушать', icon: 'play', title: 'Прослушать запись этого разбора', disabled: disabled || player?.loading === true,
      onClick: () => recorder.openPlayer({ sessionId: job.sessionId, name: job.recording, owner: job.owner, origin: 'card' })
    }))
  }
  if ((state === 'queued' || state === 'running') && view.cancel_requested !== true) {
    buttons.push(action({ label: 'Отменить разбор', icon: 'stop-circle', title: 'Отменить разбор этой записи', disabled, onClick: () => recorder.cancelJob() }))
  }
  if ((state === 'failed' || state === 'cancelled') && !view.staging_removed_at) {
    buttons.push(action({ label: 'Повторить разбор', icon: 'debug-restart', title: 'Поставить эту запись на разбор ещё раз', disabled, onClick: () => recorder.retryJob() }))
  }
  if (job.result) {
    buttons.push(action({ label: 'Открыть отчёт', icon: 'output', title: 'Открыть полный отчёт разбора', disabled, onClick: () => recorder.openReport() }))
  }
  if (job.stopped) {
    buttons.push(action({ label: 'Обновить', icon: 'refresh', title: 'Запросить статус разбора снова', disabled, onClick: () => recorder.refresh() }))
  }
  if (job.final || job.stopped) {
    buttons.push(action({ label: 'Скрыть', icon: 'close', title: 'Скрыть карточку разбора', disabled, onClick: () => recorder.dismiss() }))
  }
  if (buttons.length) children.push(jsx('div', { className: 'sound-scout-card__actions', children: buttons }, 'actions'))
  if (job.report?.open) children.push(jsx(ReportDialog, { recorder, job }, 'report'))
  return jsxs('section', { className: 'sound-scout-card', 'aria-label': 'Разбор Sound Scout', children })
}

// Without useComposerSession the recorder cannot tell its chat, so it offers no Listen.
const composerSession = () => typeof useComposerSession === 'function' ? useComposerSession() : null

export function RecorderControls({ recorder }) {
  const state = useValue(recorder.state)
  const elapsed = useValue(recorder.elapsed)
  const sent = useValue(recorder.sent)
  const player = useValue(recorder.player)
  const stored = composerSession()?.storedSessionId ?? null
  const { phase } = state
  const busy = phase === 'sending' || phase === 'starting'
  const children = [jsx('style', { children: STYLE }, 'style')]
  if (state.error) {
    children.push(jsx('span', { role: 'alert', title: state.error, className: 'sound-scout-recorder__error', children: state.error }, 'error'))
  }
  // One dialog per window: only the composer of the chat of the player renders it.
  if (player?.url && stored !== null && stored === player.sessionId) children.push(jsx(PlayerDialog, { recorder, player }, 'player'))
  if (phase === 'idle') {
    children.push(action({ label: 'Записать', icon: 'mic', title: 'Записать голосовое сообщение', onClick: () => recorder.start() }))
    const last = stored === null ? null : sent[stored]
    if (last) {
      children.push(action({
        label: 'Прослушать', icon: 'play', title: 'Прослушать последнюю отправленную запись этого чата', disabled: player?.loading === true,
        onClick: () => recorder.openPlayer({ sessionId: stored, name: last.name, owner: last.owner, origin: 'recorder' })
      }))
      if (player?.loading && player.origin === 'recorder') {
        children.push(jsx('span', { role: 'status', className: 'sound-scout-recorder__status', children: 'Загрузка записи…' }, 'loading'))
      }
    }
    return jsxs('div', { className: 'sound-scout-recorder', 'data-phase': phase, children })
  }
  const status = phase === 'recording' ? `${clock(elapsed)} / ${clock(state.limitSeconds)}`
    : phase === 'starting' ? 'Микрофон…'
    : phase === 'sending' ? 'Отправка…'
    : state.restored ? 'Неотправленная запись'
    : 'К отправке'
  children.push(jsxs('span', { role: 'status', className: 'sound-scout-recorder__status', children: [
    phase === 'recording' && jsx('span', { className: 'sound-scout-recorder__dot', 'aria-hidden': true }),
    jsx('span', { children: status }),
    phase === 'recording' && jsx(SignalMeter, { recorder })
  ] }, 'status'))
  if (state.unprotected) {
    children.push(jsx('span', { role: 'note', className: 'sound-scout-recorder__job', children: 'Запись не защищена от перезагрузки окна' }, 'unprotected'))
  }
  children.push(action({ label: 'Отправить', icon: 'send', title: 'Отправить запись в чат', disabled: busy, onClick: () => recorder.send() }))
  children.push(action({
    label: 'Отправить и разобрать', icon: 'pulse',
    title: state.canAnalyze ? 'Отправить запись в чат и поставить её на разбор' : `Разбор недоступен: ${state.intakeError}`,
    disabled: busy || !state.canAnalyze, onClick: () => recorder.send({ analyze: true })
  }))
  children.push(state.restored
    ? action({ label: 'Удалить', icon: 'trash', title: 'Удалить сохранённую запись', disabled: busy, onClick: () => recorder.cancel() })
    : action({ label: 'Отмена', icon: 'close', title: 'Отменить запись', disabled: busy, onClick: () => recorder.cancel() }))
  return jsxs('div', { className: 'sound-scout-recorder', 'data-phase': phase, children })
}

function supportsRecorder(ctx) {
  const state = host.state
  return typeof host.profileRoutes === 'function' && typeof host.retainProfile === 'function' &&
    typeof host.requestProfile === 'function' && typeof state?.focusedStoredSessionId?.get === 'function' &&
    typeof state?.focusedSessionOwner?.get === 'function' && typeof ctx.rest === 'function'
}

function supportsCard() {
  return typeof useComposerSession === 'function' && Boolean(COMPOSER_AREAS.top)
}

export default {
  id: 'sound-scout',
  name: 'Sound Scout',
  register(ctx) {
    const health = async () => {
      const title = 'Sound Scout: health'
      try {
        const answer = await ctx.rest('/health')
        const ready = answer.ok && answer.intake.ready
        host.notify({ kind: ready ? 'info' : 'warning', title, message: describeHealth(answer) })
      } catch (error) {
        host.notify({ kind: 'error', title, message: String(error) })
      }
    }

    ctx.register({
      id: 'health',
      area: PALETTE_AREA,
      data: {
        id: 'sound-scout.health',
        label: 'Sound Scout: health',
        keywords: ['sound', 'scout', 'health'],
        run: () => void health()
      }
    })

    if (!supportsRecorder(ctx)) {
      host.notify({ kind: 'error', message: 'Sound Scout: эта версия Hermes Desktop не даёт рекордеру нужных вызовов SDK. Запись недоступна.' })
      return
    }
    const recorder = new Recorder(host, ctx)
    ctx.register({ id: 'record', area: COMPOSER_AREAS.actions, render: () => jsx(RecorderControls, { recorder }) })
    recorder.restore().catch(() => {})
    if (supportsCard()) ctx.register({ id: 'job-card', area: COMPOSER_AREAS.top, render: () => jsx(JobCard, { recorder }) })
    const beforeUnload = event => {
      if (recorder.state.get().phase === 'idle') return
      event.preventDefault()
      event.returnValue = ''
    }
    globalThis.addEventListener?.('beforeunload', beforeUnload)
    ctx.onDispose(() => {
      globalThis.removeEventListener?.('beforeunload', beforeUnload)
      recorder.dispose()
    })
  }
}
