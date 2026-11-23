// Desktop half of the Sound Scout package (#101, decision 0013 § 3.2–3.4): the recorder of the
// composer (#103), the health command of the palette and the sound-scout-report transcript
// directive (its card is #108). The recorder sends by the stock path Spike A chose (decision 0016).
import { atom, host, Button, Codicon, COMPOSER_AREAS, PALETTE_AREA, TRANSCRIPT_DIRECTIVE_AREA, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

const DIRECTIVE = 'sound-scout-report'
const SPIKE_JOB = /^spike-[0-9a-f]{32}$/
const JOB_ID = /^[0-9a-f]{32}$/
const FOREGROUND = { spawnPriority: 'foreground' }
const ATTACH_TIMEOUT_MS = 120000
const UPLOAD_TIMEOUT_MS = 120000
const STATUS_TIMEOUT_MS = 10000
const HEALTH_TIMEOUT_MS = 10000
const WAV_HEADER_BYTES = 44

function ReportCard({ job }) {
  return jsx('div', {
    className: 'rounded-md border px-3 py-2 text-sm',
    children: `Sound Scout: карточка отчёта ${job}`
  })
}

function describeHealth(answer) {
  const budgets = answer.budgets
    ? `до ${answer.budgets.max_duration_seconds} с, до ${answer.budgets.max_input_bytes} байт`
    : 'бюджеты недоступны'
  const intake = answer.intake.ready
    ? 'приём записей готов'
    : `приём записей: ${answer.intake.error}: ${answer.intake.detail}`
  return `pid=${answer.pid} session_owner=${answer.session_owner}; ${budgets}; ${intake}`
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

const positive = value => Number.isInteger(value) && value > 0
const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`

function describeJob(job, result) {
  if (job.state === 'queued') return { text: 'Разбор: в очереди', final: false }
  if (job.state === 'running') {
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

export class Recorder {
  constructor(host, ctx, {
    sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
    now = () => Date.now(),
    pollEveryMs = 3000,
    maxPollFailures = 10,
    random = n => crypto.getRandomValues(new Uint8Array(n))
  } = {}) {
    this.host = host
    this.ctx = ctx
    this.sleep = sleep
    this.now = now
    this.pollEveryMs = pollEveryMs
    this.maxPollFailures = maxPollFailures
    this.random = random
    this.state = atom({ phase: 'idle', error: '', startedAt: 0, limitSeconds: 0, canAnalyze: false, intakeError: '' })
    this.elapsed = atom(0)
    this.level = atom(0)
    this.job = atom(null)
    this.levelAt = -Infinity
    this.chunks = []
    this.generation = 0
    this.pollGeneration = 0
    this.polling = null
    this.pending = null
    this.target = null
    this.budget = null
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
    this.update({ phase: 'starting', error: '' })
    if (!target.stored) {
      this.update({ phase: 'idle', error: 'Открой существующий чат перед записью.' })
      return
    }
    let health
    try {
      health = await this.readBudget()
    } catch (error) {
      if (generation !== this.generation) return
      this.update({ phase: 'idle', error: `Sound Scout не сообщил лимит записи: ${error.message}.` })
      return
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
    this.generation++
    this.releaseCapture()
    this.chunks = []
    this.pending = null
    this.update({ phase: 'idle', error: '' })
  }

  dispose() {
    this.pollGeneration++
    this.cancel()
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
    this.pending = { blob, dataUrl: null, token, owner: null, attach: null, submitted: false, job: null }
    this.update({ phase: 'ready' })
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
          }
          message = 'Чат не принял сообщение. Аудио уже загружено, повтори отправку.'
          const receipt = await this.host.requestProfile(route, 'prompt.submit', {
            session_id: resumed.session_id, text: pending.attach.ref_text, submission_id: `sound-scout-${pending.token}`
          })
          if (generation !== this.generation) return
          if (typeof receipt?.admission_id !== 'string' || !receipt.admission_id) throw new Error('Чат не подтвердил сообщение')
          pending.submitted = true
        } finally { release?.() }
      }
      if (analyze && !pending.job) {
        message = null
        await this.submitJob(pending, target, generation)
        if (generation !== this.generation) return
      }
      this.pending = null
      this.update({ phase: 'idle' })
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
    this.track(answer.job_id, owner)
  }

  onRoute(owner) {
    const state = this.host.state
    return state.connectionId?.get?.() === owner.connectionId && state.profile?.get?.() === owner.profile
  }

  track(jobId, owner) {
    const poll = ++this.pollGeneration
    this.job.set({ jobId, text: 'Разбор: принят', final: false })
    this.polling = this.poll(jobId, owner, poll)
  }

  async poll(jobId, owner, poll) {
    let failures = 0
    while (poll === this.pollGeneration) {
      await this.sleep(this.pollEveryMs)
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
      const seen = answer?.ok === true && answer.job ? describeJob(answer.job, answer.result) : null
      if (seen) {
        failures = 0
        this.job.set({ jobId, text: seen.text, final: seen.final })
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
        this.job.set({ jobId, text, final: true })
        this.host.notify({ kind: 'warning', message: text })
        return
      }
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
      jsx(Codicon, { name: icon, size: 14 }), jsx('span', { children: label })
    ] })
  }, label)
}

const STYLE = `
  .sound-scout-recorder { position:relative; display:inline-flex; align-items:center; gap:4px; max-width:100%; min-height:28px; }
  .sound-scout-recorder__status { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; color:var(--ui-text-secondary); font-variant-numeric:tabular-nums; }
  .sound-scout-recorder__dot { width:6px; height:6px; border-radius:50%; background:var(--ui-red); }
  .sound-scout-recorder__meter { display:inline-flex; align-items:center; gap:2px; width:14px; height:12px; color:var(--ui-text-secondary); }
  .sound-scout-recorder__meter span { display:block; width:2px; border-radius:1px; background:currentColor; }
  .sound-scout-recorder__label { display:inline-flex; align-items:center; gap:5px; }
  .sound-scout-recorder__job { color:var(--ui-text-secondary); font-size:12px; white-space:nowrap; }
  .sound-scout-recorder__error { position:absolute; bottom:calc(100% + 6px); left:0; z-index:10; width:max-content; max-width:min(260px,80vw); padding:6px 8px; border:1px solid var(--ui-stroke-secondary); border-radius:4px; background:var(--ui-bg-quaternary); color:var(--ui-text-secondary); font-size:12px; }
`

function RecorderControls({ recorder }) {
  const state = useValue(recorder.state)
  const elapsed = useValue(recorder.elapsed)
  const job = useValue(recorder.job)
  const { phase } = state
  const busy = phase === 'sending' || phase === 'starting'
  const children = [jsx('style', { children: STYLE }, 'style')]
  if (state.error) {
    children.push(jsx('span', { role: 'alert', title: state.error, className: 'sound-scout-recorder__error', children: state.error }, 'error'))
  }
  if (phase === 'idle') {
    children.push(action({ label: 'Записать', icon: 'mic', title: 'Записать голосовое сообщение', onClick: () => recorder.start() }))
    if (job) children.push(jsx('span', { role: 'status', className: 'sound-scout-recorder__job', children: job.text }, 'job'))
    return jsxs('div', { className: 'sound-scout-recorder', 'data-phase': phase, children })
  }
  const status = phase === 'recording' ? `${clock(elapsed)} / ${clock(state.limitSeconds)}`
    : phase === 'starting' ? 'Микрофон…'
    : phase === 'sending' ? 'Отправка…'
    : 'К отправке'
  children.push(jsxs('span', { role: 'status', className: 'sound-scout-recorder__status', children: [
    phase === 'recording' && jsx('span', { className: 'sound-scout-recorder__dot', 'aria-hidden': true }),
    jsx('span', { children: status }),
    phase === 'recording' && jsx(SignalMeter, { recorder })
  ] }, 'status'))
  children.push(action({ label: 'Отправить', icon: 'send', title: 'Отправить запись в чат', disabled: busy, onClick: () => recorder.send() }))
  children.push(action({
    label: 'Отправить и разобрать', icon: 'pulse',
    title: state.canAnalyze ? 'Отправить запись в чат и поставить её на разбор' : `Разбор недоступен: ${state.intakeError}`,
    disabled: busy || !state.canAnalyze, onClick: () => recorder.send({ analyze: true })
  }))
  children.push(action({ label: 'Отмена', icon: 'close', title: 'Отменить запись', disabled: busy, onClick: () => recorder.cancel() }))
  return jsxs('div', { className: 'sound-scout-recorder', 'data-phase': phase, children })
}

function supportsRecorder(ctx) {
  const state = host.state
  return typeof host.profileRoutes === 'function' && typeof host.retainProfile === 'function' &&
    typeof host.requestProfile === 'function' && typeof state?.focusedStoredSessionId?.get === 'function' &&
    typeof state?.focusedSessionOwner?.get === 'function' && typeof ctx.rest === 'function'
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

    ctx.registerMany([
      {
        id: 'health',
        area: PALETTE_AREA,
        data: {
          id: 'sound-scout.health',
          label: 'Sound Scout: health',
          keywords: ['sound', 'scout', 'health'],
          run: () => void health()
        }
      },
      {
        id: 'report-directive',
        area: TRANSCRIPT_DIRECTIVE_AREA,
        data: {
          name: DIRECTIVE,
          render: ({ attrs }) => {
            const job = attrs && attrs.job
            return typeof job === 'string' && SPIKE_JOB.test(job) ? jsx(ReportCard, { job }) : null
          }
        }
      }
    ])

    if (!supportsRecorder(ctx)) {
      host.notify({ kind: 'error', message: 'Sound Scout: эта версия Hermes Desktop не даёт рекордеру нужных вызовов SDK. Запись недоступна.' })
      return
    }
    const recorder = new Recorder(host, ctx)
    ctx.register({ id: 'record', area: COMPOSER_AREAS.actions, render: () => jsx(RecorderControls, { recorder }) })
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
