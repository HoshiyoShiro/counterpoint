# Counterpoint

Two AI models hold a real conversation with each other about a question, then a third
model that never took part writes the verdict. Runs across eight providers — OpenRouter,
Groq, Mistral, Cerebras, Cloudflare Workers AI, Kilo, Ollama Cloud, and a local Ollama
daemon — so one vendor's rate limit cannot stall a run. Zero dependencies, stdlib only.

## Setup

```bash
cp .env.example .env      # then paste your keys into .env
```

`.env` is gitignored. Any one key is enough; more keys mean more headroom.

| Provider | Key | Free models |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` — https://openrouter.ai/keys | everything tagged `:free` |
| Groq | `GROQ_API_KEY` — https://console.groq.com/keys | the zero-priced ones (see below) |
| Mistral | `MISTRAL_API_KEY` — https://console.mistral.ai | all chat models, free on the Experiment tier |
| Cerebras | `CEREBRAS_API_KEY` — https://cloud.cerebras.ai | the whole catalogue (`gemma-4-31b`, `gpt-oss-120b`) |
| Cloudflare | `CLOUDFLARE_API_TOKEN` **and** `CLOUDFLARE_ACCOUNT_ID` — https://dash.cloudflare.com/profile/api-tokens | every Text Generation model on Workers AI |
| Kilo | `KILO_API_KEY` — https://kilo.ai | the zero-priced ones (14 at last check) |
| Ollama Cloud | `OLLAMA_API_KEY` — https://ollama.com/settings/keys | 6 of 19; the rest need a subscription |
| Ollama (local) | *none* — just run `ollama serve` | whatever you have pulled |

### What counts as free

- **OpenRouter** marks free models with a `:free` suffix, and only those are listed.
  Its free tier has a hard **daily request cap** — once hit, nothing works until it
  resets, which is why OpenRouter sits last in the fallback chain.
- **Groq** publishes per-token pricing, so "free" means `prompt == completion == 0`:
  `groq/compound`, `groq/compound-mini`, `allam-2-7b`. Set `GROQ_ALLOW_PAID=1` to list
  the metered ones too. Note `compound` internally routes to gpt-oss-120b and shares
  that model's rate limit.
- **Mistral** publishes no per-model pricing — on the free Experiment tier every model
  is free but rate-limited, and on a paid plan every model is metered. So the filter is
  chat-capability, not price. Only `-latest` aliases are listed; `MISTRAL_ALL_MODELS=1`
  shows all 56.
- **Cerebras** exposes neither pricing nor capability metadata and lists only two chat
  models, so both are offered as-is. Note that a Cerebras account without free quota
  answers every inference call with `402 payment_required` even though the key is valid
  and `/models` works — see below for how that is handled.
- **Kilo** is an OpenRouter-shaped gateway (`https://kilo.ai/api/openrouter`) running on
  its own account and its own quota. That overlap is the point: the same model can be
  reached through either vendor, so Kilo keeps working after OpenRouter's daily cap is
  spent. It publishes OpenRouter's pricing block, so `prompt == completion == 0` is the
  filter. The `openrouter/*` and `kilo-auto/*` router pseudo-models are excluded — they
  answer with empty content rather than a usable turn.
- **Ollama Cloud** lists its catalogue anonymously but 401s on inference without a key.
  On a keyed free account 6 of the 19 models answer — `gpt-oss:120b`, `gpt-oss:20b`,
  `gemma4:31b`, `minimax-m3`, `nemotron-3-nano:30b`, `nemotron-3-super` — and the rest
  return `403 this model requires a subscription`. There is no metadata distinguishing
  them, so all 19 are listed and the gated ones fail per model with a clear message.
- **Ollama (local)** needs no key at all and has no quota to exhaust, which earns it the
  last slot in the fallback chain: when every hosted free tier is dry, the daemon on your
  own machine still answers. Detected by a cached 0.6s TCP probe of `OLLAMA_HOST`
  (default `http://127.0.0.1:11434`); if nothing is listening the provider simply does
  not appear.
- **Cloudflare** bills every Workers AI model out of one shared daily neuron allowance
  rather than pricing them individually, so the free tier is the account quota, not a
  subset of models. The filter is therefore task-based: Text Generation only.
  **It needs two values, not one** — the token alone is not enough, because every
  Workers AI path is scoped to an account (`/accounts/{id}/ai/v1/...`). Find the account
  id on any domain's overview page in the dash, or under Workers & Pages. An API token
  cannot look it up: `/accounts` returns an empty list for account-scoped tokens.

## Web UI

```bash
python server.py          # opens http://127.0.0.1:8765
```

Topic box, mode, round count, and a dropdown per model (agent A, agent B, judge)
populated live from every configured provider, grouped by vendor. Turns stream in over Server-Sent Events
as they are generated — agent A on the left, agent B on the right, retry and
model-switch notices inline, judge verdict as a card at the end. Past runs are listed
in the sidebar and reload into the same view.

The server binds `127.0.0.1` only and the API key never leaves the Python process —
the browser only ever sees model *names*. `--host 0.0.0.0` would expose an unauthenticated
key-spending endpoint to your network, so don't, unless you know exactly who is on it.

Flags: `--port`, `--host`, `--no-browser`.

## CLI

```bash
# default: 4 exchanges each, collaborative mode
python debate.py "Is Rust worth it for a CRUD backend?"

# adversarial: assigned sides
python debate.py "AI agents should have write access to prod" --mode debate --rounds 6

# design review
python debate.py "Event sourcing for a booking system" --mode review

# council: one model from every configured provider, at one table
python debate.py "Is 'move fast and break things' defensible?" --council

# two seats per provider (up to 12 members)
python debate.py "topic" --council --per-provider 2

# pick your own models — the prefix picks the provider
python debate.py "topic" --model-a groq:groq/compound \
                         --model-b mistral:ministral-8b-latest

# what free models exist right now, per provider
python debate.py --list-models
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--rounds` | 4 (20 in council) | rounds; `0` means let them decide |
| `--min-rounds` | 2 | convergence ignored before this round |
| `--mode` | `discuss` | `discuss` \| `debate` \| `review` \| `council` |
| `--council` | off | shorthand for `--mode council` |
| `--per-provider` | 1 | council seats per provider (1 or 2) |
| `--model-a` | `mistral:mistral-small-latest` | agent A |
| `--model-b` | `groq:groq/compound-mini` | agent B |
| `--model-judge` | `mistral:mistral-medium-latest` | writes the synthesis |
| `--names` | persona defaults | e.g. `--names "Alice,Bob"` |
| `--temperature` | 0.8 | |
| `--max-tokens` | 500 | per turn |
| `--no-fallback` | off | fail instead of switching model on rate limit |

## How it works

Each agent keeps its **own** message list. Its own turns are `assistant`, the other
agent's turns arrive as `user` prefixed with the speaker's name — so each model
genuinely experiences the other as an interlocutor rather than replaying a script.

- **Convergence**: an agent ends a turn with `[CONVERGED]` when it has nothing left to
  add. The run stops only when *both* do it, and never before `--min-rounds`.
- **Synthesis**: a third model that did not debate writes Position / Agreed /
  Unresolved / Confidence, so no model grades its own argument.
- **Transcripts**: saved to `transcripts/` as both `.md` and `.json`.

## Council mode

`--council` seats one model from **every configured provider** — or two with
`--per-provider 2` — at a single table and lets them talk until they agree they are
finished.

- **Round-robin.** Each member speaks once per round and hears every other member's
  turn as an incoming message, exactly as in two-agent mode.
- **They decide when to stop.** There is no fixed round count. A member ends a turn with
  `[CONVERGED]` when it thinks the *group* is done, and the discussion ends only once
  every member still seated does so in the same round. `--min-rounds` (default 2) stops
  them agreeing instantly.
- **Capped at 20 rounds** as a backstop, so a council that never agrees still terminates
  and gets a verdict.
- **Distinct personas.** Each seat gets its own thinking style — adversary, pragmatist,
  historian, simplifier, empiricist — so eight models do not produce eight of the same
  answer. Twelve personas exist, which is also the hard seat limit: two members answering
  to the same name would make every reply ambiguous.
- **A dead seat is replaced, not fatal.** If a member's model fails, it is re-seated from
  a bench of every available model, preferring a *different* vendor and skipping models
  other seats already use. If no substitute exists and more than two members remain, that
  seat leaves the council and the discussion continues without it.

## Substitutions

Any model whose provider is unconfigured or failing is replaced rather than aborting the
run — including the ones you picked explicitly:

- **Before the run**, `--model-a` / `--model-b` / `--model-judge` pointing at a provider
  with no credentials are swapped for a working model and a notice explains the swap.
- **During the run**, a model that rate-limits or dies falls through the fallback chain,
  which alternates vendors deliberately.
- **In council**, substitutes are drawn from the whole bench and never duplicate a model
  another seat is already speaking with.

## Model ids

Every model id is namespaced by provider:

    mistral:mistral-small-latest
    groq:groq/compound-mini
    cerebras:gpt-oss-120b
    cloudflare:@cf/meta/llama-3.1-8b-instruct
    kilo:tencent/hy3:free
    ollama-cloud:gpt-oss:120b
    ollama:llama3.2:latest
    openrouter:z-ai/glm-5.2:free
    z-ai/glm-5.2:free              # no prefix = OpenRouter, for back-compat

## Free-tier realities

Free models return HTTP 429 constantly. The defence is vendor diversity: the fallback
chain in `debate.py` alternates providers, because a jump to a different vendor is far
likelier to succeed than another wait on the same saturated pool.

- Each call retries 4 times with exponential backoff (2s → 16s).
- **Except** when the 429 says the daily or monthly quota is gone — that will not clear
  during a run, so the call fails immediately and the next provider is tried instead of
  burning 30s on a doomed backoff ladder.
- If a model stays down, that agent falls through `FALLBACKS` and keeps going, printing
  `switched Ada: … -> …`. Disable with `--no-fallback`.
- A `401`/`402`/`403` is normally account-level: every model behind that key would fail
  identically, so the whole provider is disabled for the rest of the process rather than
  being retried once per turn. **Unless** the body says it is about one model — "this
  model requires a subscription", "upgrade", "not available on your plan" — in which case
  only that model fails and its working siblings on the same key stay usable.
- An empty `content` alongside a populated `reasoning` field means the model spent its
  whole budget thinking and never reached an answer. That is treated as a failure rather
  than returned, so raw chain-of-thought never lands in a debate; Ollama Cloud's reasoning
  models also get a raised token floor so it rarely happens.
- Fallback entries whose provider has no credentials configured are dropped before the
  run starts, so an unconfigured provider costs nothing per turn.
- Params a model rejects (`reasoning`, `reasoning_format`) are dropped and remembered
  per model, so the retry is paid once per process, not once per turn. The match is
  case-insensitive, which also catches the inverse complaint — "Reasoning is mandatory
  for this endpoint and cannot be disabled" — where the fix is likewise to stop sending
  our override.
- Reasoning models leak their scratchpad into `content`, so requests send
  `reasoning: {enabled: false}` (OpenRouter) or `reasoning_format: hidden` (Groq), and
  replies are additionally scrubbed of `<think>` blocks, `**Reasoning**` preambles, and
  `Name:` prefixes.
- A degenerate reply (under 40 chars) triggers one retry, then a model switch.

## Files

- `providers.py` — provider registry, model routing, retries, `.env` loader
- `debate.py` — personas, turn loop, convergence, synthesis, transcript writing
- `server.py` — localhost HTTP + SSE server for the web UI
- `static/index.html` — the whole frontend, no build step, no dependencies

`debate.converse(args)` is a generator yielding `meta` / `turn-start` / `turn` /
`notice` / `converged` / `synthesis` / `saved` / `error` events. The CLI renderer and
the SSE endpoint are both thin consumers of it, so any behaviour change lands in both.

## API

| Route | Returns |
|---|---|
| `GET /api/config` | free model ids grouped by provider, modes, defaults |
| `GET /api/history` | saved runs (name, topic, mode, date, turn count) |
| `GET /api/history/<name>` | one saved run as JSON |
| `GET /api/stream?topic=…` | SSE event stream of a live run |

## Adding a provider

Subclass `Provider` in `providers.py` with a `name`, `base` URL, and `key_env`, then add
it to the tuple at the bottom of the file. Override `payload()` for vendor-specific
params and `filter_models()` for what counts as free. Anything speaking the OpenAI
chat-completions shape needs nothing else.

`CloudflareProvider` is the worked example of a provider that does not fit cleanly: its
`base` is a property because the URL embeds the account id, `available` requires two env
vars, and `list_models` is overridden because the catalogue lives at a non-OpenAI path.

## License

[MIT](LICENSE).
