import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs'
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'

const LAYERS = 16,
  HEADS = 10,
  HEAD_DIM = 64,
  VOCAB = 32768,
  CONTEXT = 2048

// ---- tokenizer ----
// Port of the C++ Tokenizer: same pretokenizer pattern, then BPE merges applied lowest rank first.
const PRETOKENIZE =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

class Tokenizer {
  constructor (json) {
    Object.assign(this, {
      eos: json.eos,
      user: json.user,
      assistant: json.assistant,
      end: json.end
    })
    this.tokenBytes = json.tokens.map(b64 =>
      Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    )
    this.rank = new Map() // pair (a * 65536 + b) -> [priority, merged token]
    for (let i = 0; i < json.merges.length / 3; i++) {
      this.rank.set(json.merges[3 * i] * 65536 + json.merges[3 * i + 1], [
        i,
        json.merges[3 * i + 2]
      ])
    }
    this.encoder = new TextEncoder()
    this.decoder = new TextDecoder('utf-8')
  }

  encodeWord (word) {
    let seq = Array.from(this.encoder.encode(word)) // the word's UTF-8 bytes
    while (seq.length > 1) {
      let bestRank = Infinity,
        bestKey = -1
      for (let i = 0; i + 1 < seq.length; i++) {
        const entry = this.rank.get(seq[i] * 65536 + seq[i + 1])
        if (entry && entry[0] < bestRank) {
          bestRank = entry[0]
          bestKey = seq[i] * 65536 + seq[i + 1]
        }
      }
      if (bestKey < 0) break
      const first = Math.floor(bestKey / 65536),
        second = bestKey % 65536,
        merged = this.rank.get(bestKey)[1]
      const out = []
      for (let i = 0; i < seq.length; i++) {
        if (i + 1 < seq.length && seq[i] === first && seq[i + 1] === second) {
          out.push(merged)
          i++
        } else out.push(seq[i])
      }
      seq = out
    }
    return seq
  }

  encode (text) {
    const ids = []
    for (const match of text.matchAll(PRETOKENIZE))
      for (const id of this.encodeWord(match[0])) ids.push(id)
    return ids
  }

  chatPrompt (text) {
    return [this.user, ...this.encode(text), this.end, this.assistant]
  }

  // streaming decode: a character split across two tokens is held back until it is complete
  decodeToken (id) {
    return this.decoder.decode(this.tokenBytes[id], { stream: true })
  }
}
// ---- end tokenizer ----

// temperature -> top-p -> sample, with a repetition penalty on tokens already in the context
function sample (logits, context, { temperature, topP, penalty }) {
  const scores = Float32Array.from(logits)
  for (const t of new Set(context))
    scores[t] = scores[t] > 0 ? scores[t] / penalty : scores[t] * penalty // repetition penalty
  let best = 0
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i
  if (temperature < 0.01) return best // greedy
  const probs = new Float32Array(scores.length)
  let sum = 0
  for (let i = 0; i < scores.length; i++) {
    probs[i] = Math.exp((scores[i] - scores[best]) / temperature)
    sum += probs[i]
  } // temperature
  const candidates = []
  for (let i = 0; i < probs.length; i++)
    if (probs[i] > sum * 1e-6) candidates.push(i) // drop the negligible tail before sorting
  candidates.sort((a, b) => probs[b] - probs[a])
  let kept = 0,
    cumulative = 0 // top-p: smallest set reaching topP
  while (kept < candidates.length && cumulative < topP * sum)
    cumulative += probs[candidates[kept++]]
  let r = Math.random() * cumulative
  for (let i = 0; i < kept; i++) {
    r -= probs[candidates[i]]
    if (r <= 0) return candidates[i]
  }
  return candidates[kept - 1]
}

const log = document.getElementById('log'),
  statusEl = document.getElementById('status')
const promptEl = document.getElementById('prompt'),
  sendEl = document.getElementById('send'),
  stopEl = document.getElementById('stop')
const setStatus = text => {
  statusEl.textContent = text
}
const addMessage = (cls, text) => {
  const div = document.createElement('div')
  div.className = 'msg ' + cls
  div.textContent = text
  log.appendChild(div)
  div.scrollIntoView({ block: 'end' })
  return div
}

async function download (url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`)
  const total = +response.headers.get('content-length') || 0
  const reader = response.body.getReader(),
    chunks = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    if (total)
      setStatus(
        `Downloading ${label}: ${Math.round((100 * received) / total)}%`
      )
  }
  const bytes = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.length
  }
  return bytes
}

let tokenizer, session
const useGpu = !!navigator.gpu
const pastNames = [],
  presentNames = []
for (let i = 0; i < LAYERS; i++)
  for (const kv of ['k', 'v']) {
    pastNames.push(`past_${i}_${kv}`)
    presentNames.push(`present_${i}_${kv}`)
  }

async function load () {
  const tokenizerResponse = await fetch('tokenizer.json')
  if (!tokenizerResponse.ok)
    throw new Error(
      `tokenizer.json: HTTP ${tokenizerResponse.status} (is it in the folder the server is serving?)`
    )
  tokenizer = new Tokenizer(await tokenizerResponse.json())
  // const model = await download('model.onnx', 'model')
  const model = await download("https://huggingface.co/oliverhardi/zero/resolve/main/model.onnx", "model")
  setStatus(
    `Starting ${
      useGpu ? 'WebGPU' : 'WASM (no WebGPU in this browser: this will be slow)'
    }…`
  )
  session = await ort.InferenceSession.create(model, {
    executionProviders: useGpu ? ['webgpu'] : ['wasm'],
    // keep the KV cache on the GPU between steps instead of copying it to the CPU and back
    preferredOutputLocation: useGpu
      ? Object.fromEntries(presentNames.map(n => [n, 'gpu-buffer']))
      : undefined
  })
  setStatus('Ready.')
  promptEl.disabled = sendEl.disabled = false
  promptEl.focus()
}

let stopRequested = false
let turns = [] // one token array per finished exchange
document.getElementById('reset').onclick = () => {
  turns = []
  log.replaceChildren()
  setStatus('Ready.')
}
stopEl.onclick = () => {
  stopRequested = true
}

// context = the whole conversation so far as token ids (history + new prompt); the generated tokens are appended to it
async function generate (context, output, options) {
  let input = context.slice()
  let past = Object.fromEntries(
    pastNames.map(n => [
      n,
      new ort.Tensor('float32', new Float32Array(0), [1, HEADS, 0, HEAD_DIM])
    ])
  )
  let text = '',
    started = performance.now(),
    generated = 0
  tokenizer.decoder = new TextDecoder('utf-8')
  try {
    while (
      generated < options.maxNew &&
      context.length < CONTEXT &&
      !stopRequested
    ) {
      const feeds = {
        ids: new ort.Tensor('int64', BigInt64Array.from(input, BigInt), [
          1,
          input.length
        ]),
        ...past
      }
      const out = await session.run(feeds)
      for (const t of Object.values(past)) t.dispose() // the old cache is replaced by the new one
      past = Object.fromEntries(
        pastNames.map((n, i) => [n, out[presentNames[i]]])
      )

      const token = sample(out.logits.data, context, options)
      if (token === tokenizer.end || token === tokenizer.eos) break
      context.push(token)
      input = [token]
      generated++
      text += tokenizer.decodeToken(token)
      output.textContent = text
      output.scrollIntoView({ block: 'end' })
      setStatus(
        `${(generated / ((performance.now() - started) / 1000)).toFixed(
          1
        )} tokens/s`
      )
    }
  } finally {
    for (const t of Object.values(past)) t.dispose()
  }
}

promptEl.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); document.getElementById("form").requestSubmit(); }
});

document.getElementById('form').onsubmit = async event => {
  event.preventDefault()
  const text = promptEl.value.trim()
  if (!text) return
  promptEl.value = ''
  addMessage('user', text)
  const output = addMessage('assistant', '…')
  const value = id => +document.getElementById(id).value
  const options = {
    temperature: value('temperature'),
    topP: value('topP'),
    penalty: value('penalty'),
    maxNew: value('maxNew')
  }
  promptEl.disabled = sendEl.disabled = true
  stopEl.hidden = false
  stopRequested = false

  // The model only knows positions 0..2047: when the conversation plus a reply no longer fits, start a fresh context.
  const prompt = tokenizer.chatPrompt(text)
  if (turns.flat().length + prompt.length + options.maxNew > CONTEXT) {
    turns = []
    log.insertBefore(
      Object.assign(document.createElement('div'), {
        className: 'msg',
        textContent: 'Context full: earlier messages are forgotten from here.'
      }),
      output.previousSibling
    )
  }
  const history = turns.flat()
  const context = [...history, ...prompt]
  try {
    await generate(context, output, options)
    turns.push([...context.slice(history.length), tokenizer.end]) // this turn: prompt + reply + <|end|>, as in the SFT data
  } catch (error) {
    output.textContent += '\n[error: ' + error.message + ']'
    console.error(error)
  }
  stopEl.hidden = true
  promptEl.disabled = sendEl.disabled = false
  promptEl.focus()
}

load().catch(error => {
  setStatus('Failed to load: ' + error.message)
  console.error(error)
})
