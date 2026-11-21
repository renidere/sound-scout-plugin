// Loads a Desktop plugin file in a vm context with a fake @hermes/plugin-sdk and react/jsx-runtime,
// after tests/load-plugin.mjs of renidere/hermes-voice-note v0.4.2. Any other module is an error.
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'

export const PROBE_PATH = new URL('../../hermes/probe/sound-scout-probe-a/plugin.js', import.meta.url)
export const RECORDER_PATH = new URL('../../hermes/plugin/sound-scout/desktop/plugin.js', import.meta.url)

class FileReader {
  readAsDataURL(blob) {
    blob.arrayBuffer().then(buffer => {
      this.result = `data:${blob.type};base64,${Buffer.from(buffer).toString('base64')}`
      this.onload?.()
    }).catch(error => {
      this.error = error
      this.onerror?.()
    })
  }
}

function atom(initial) {
  let value = initial
  const listeners = new Set()
  return {
    get: () => value,
    set(next) {
      value = next
      for (const listener of listeners) listener(next)
    },
    subscribe(listener) {
      listeners.add(listener)
      listener(value)
      return () => listeners.delete(listener)
    }
  }
}

const element = (type, props) => ({ type, props })

export async function loadPlugin({ path = PROBE_PATH, sdk = {}, globals = {}, modules = {} } = {}) {
  const context = vm.createContext({
    Blob, FileReader, TextEncoder, URLSearchParams, ArrayBuffer, Uint8Array, Int16Array, Float32Array,
    DataView, crypto: globalThis.crypto, console, setTimeout, clearTimeout, setInterval, clearInterval,
    navigator: {},
    ...globals
  })
  const linked = {
    '@hermes/plugin-sdk': {
      atom,
      host: {},
      useValue: store => store.get(),
      Button: 'button',
      Codicon: 'i',
      COMPOSER_AREAS: { actions: 'composer.actions' },
      PALETTE_AREA: 'palette',
      TRANSCRIPT_DIRECTIVE_AREA: 'transcript.directive',
      ...sdk
    },
    'react/jsx-runtime': { jsx: element, jsxs: element },
    ...modules
  }
  const source = await readFile(path, 'utf8')
  const name = path.pathname.split('/').slice(-2).join('/')
  const module = new vm.SourceTextModule(source, { context, identifier: name })
  await module.link(specifier => {
    const values = linked[specifier]
    if (!values) throw new Error(`Unsupported plugin import: ${specifier}`)
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value)
    }, { context })
  })
  await module.evaluate()
  return module.namespace
}
