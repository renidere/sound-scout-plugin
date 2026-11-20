// Desktop half of the Sound Scout package (#101, decision 0013 § 3.2): the health command of the
// palette, which calls the backend through ctx.rest, and the sound-scout-report transcript
// directive (its card is #108).
import { host, PALETTE_AREA, TRANSCRIPT_DIRECTIVE_AREA } from '@hermes/plugin-sdk'
import { jsx } from 'react/jsx-runtime'

const DIRECTIVE = 'sound-scout-report'
const SPIKE_JOB = /^spike-[0-9a-f]{32}$/

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
  }
}
