// Spike B Desktop half of the Sound Scout package (#96, decision 0015): palette commands that call
// the backend through ctx.rest and the sound-scout-report transcript directive.
import { host, PALETTE_AREA, TRANSCRIPT_DIRECTIVE_AREA } from '@hermes/plugin-sdk'
import { jsx } from 'react/jsx-runtime'

const DIRECTIVE = 'sound-scout-report'
const SPIKE_JOB = /^spike-[0-9a-f]{32}$/
const DELIVERY_DELAY_S = 30

function show(title, message, ok) {
  host.notify({ kind: ok ? 'info' : 'warning', title, message })
}

function ReportCard({ job }) {
  return jsx('div', {
    className: 'rounded-md border px-3 py-2 text-sm',
    children: `Sound Scout: карточка отчёта ${job}`
  })
}

export default {
  id: 'sound-scout',
  name: 'Sound Scout (spike B)',
  register(ctx) {
    let lastToken = null

    async function call(title, request, describe) {
      try {
        const answer = await request()
        const failure = answer.error
          ? `${answer.error}: ${answer.detail}`
          : `returncode=${answer.returncode} ${answer.stderr || ''}`
        show(title, answer.ok ? describe(answer) : failure, answer.ok)
        return answer
      } catch (error) {
        host.notify({ kind: 'error', title, message: String(error) })
        return null
      }
    }

    const health = () =>
      call('Sound Scout spike: health', () => ctx.rest('/health'), a =>
        `ok=${a.ok} pid=${a.pid} session_owner=${a.session_owner}`)

    const deliver = async () => {
      const chatId = host.state.focusedStoredSessionId.get()
      if (!chatId) {
        show('Sound Scout spike: deliver', 'Нет открытого чата: заметку некуда доставить', false)
        return
      }
      const answer = await call(
        'Sound Scout spike: deliver',
        () => ctx.rest('/spike/deliver', { method: 'POST', body: { chat_id: chatId, delay_s: DELIVERY_DELAY_S } }),
        a => `token=${a.token}, доставка через ${a.delay_s} с`
      )
      if (answer && answer.ok) lastToken = answer.token
    }

    const checkDelivery = () => {
      if (!lastToken) {
        show('Sound Scout spike: delivery', 'В этом окне ещё не было доставки', false)
        return
      }
      const token = lastToken
      return call('Sound Scout spike: delivery', () => ctx.rest(`/spike/deliver/${token}`), a =>
        `token=${token} state=${a.state}${a.error ? ` error=${a.error}` : ''}`)
    }

    const startRunner = () =>
      call('Sound Scout spike: start runner', () => ctx.rest('/spike/runner', { method: 'POST' }), a =>
        `unit=${a.unit} returncode=${a.returncode}`)

    const runnerStatus = () =>
      call('Sound Scout spike: runner status', () => ctx.rest('/spike/runner'), a => {
        const p = a.properties
        return `ActiveState=${p.ActiveState} MainPID=${p.MainPID} ControlGroup=${p.ControlGroup}`
      })

    const command = (id, label, run) => ({
      id,
      area: PALETTE_AREA,
      data: { id: `sound-scout.${id}`, label, keywords: ['sound', 'scout', 'spike'], run: () => void run() }
    })

    ctx.registerMany([
      command('health', 'Sound Scout spike: health', health),
      command('deliver', 'Sound Scout spike: deliver note to this chat in 30 s', deliver),
      command('check-delivery', 'Sound Scout spike: check last delivery', checkDelivery),
      command('start-runner', 'Sound Scout spike: start runner unit', startRunner),
      command('runner-status', 'Sound Scout spike: runner unit status', runnerStatus),
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
  }
}
