// Loads hermes/probe/sound-scout-probe-a/plugin.js in a vm context with a fake @hermes/plugin-sdk,
// after tests/load-plugin.mjs of renidere/hermes-voice-note v0.4.2. Any other import is an error.
import { readFile } from 'node:fs/promises'
import vm from 'node:vm'

export const PLUGIN_PATH = new URL('../../hermes/probe/sound-scout-probe-a/plugin.js', import.meta.url)

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

export async function loadPlugin({ sdk = {}, globals = {} } = {}) {
  const context = vm.createContext({
    Blob, FileReader, TextEncoder, ArrayBuffer, Uint8Array, Float32Array, DataView,
    crypto: globalThis.crypto, console, setTimeout, clearTimeout,
    ...globals
  })
  const modules = { '@hermes/plugin-sdk': { host: {}, PALETTE_AREA: 'palette', ...sdk } }
  const source = await readFile(PLUGIN_PATH, 'utf8')
  const module = new vm.SourceTextModule(source, { context, identifier: 'sound-scout-probe-a/plugin.js' })
  await module.link(name => {
    const values = modules[name]
    if (!values) throw new Error(`Unsupported plugin import: ${name}`)
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value)
    }, { context })
  })
  await module.evaluate()
  return module.namespace
}
