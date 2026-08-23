"""Multi-provider chat client. Zero dependencies.

Every provider here speaks the OpenAI chat-completions shape, so one request
builder covers them all; only the base URL, key, and a few unsupported params
differ.

Model ids are namespaced by provider prefix:

    groq:openai/gpt-oss-120b            -> Groq
    mistral:mistral-small-latest        -> Mistral
    cerebras:gpt-oss-120b               -> Cerebras
    cloudflare:@cf/meta/llama-3.1-8b-instruct -> Cloudflare Workers AI
    ollama-cloud:gpt-oss:120b            -> Ollama Cloud
    ollama:llama3.2:latest              -> a local Ollama daemon
    openrouter:z-ai/glm-5.2:free        -> OpenRouter
    z-ai/glm-5.2:free                   -> OpenRouter (default, back-compat)

The point of several providers is that their rate limits are independent: when a
model 429s, the fallback chain can jump to a completely different vendor rather
than queueing behind the same saturated pool.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_PROVIDER = "openrouter"


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    pass


# Kept so older imports keep working.
OpenRouterError = LLMError


class Provider:
    """One vendor's OpenAI-compatible endpoint."""

    name = ""
    base = ""
    key_env = ""
    # Models that exist on the account but cannot do chat completions.
    exclude = ()

    @property
    def api_key(self) -> str:
        # Read on demand: .env is loaded after this module is imported.
        return os.environ.get(self.key_env, "").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def why_unavailable(self) -> str:
        return "" if self.available else f"no API key set ({self.key_env})"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Groq's edge rejects urllib's default UA with a 403.
            "User-Agent": "counterpoint/1.0",
        }

    def payload(self, model: str, messages: list[dict], temperature: float,
                max_tokens: int) -> dict:
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def list_models(self, timeout: int = 30) -> list[str]:
        req = urllib.request.Request(f"{self.base}/models", headers=self.headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return sorted(self.filter_models(body.get("data", [])))

    def filter_models(self, data: list[dict]) -> list[str]:
        return [m["id"] for m in data if not self.is_excluded(m.get("id", ""))]

    def is_excluded(self, model_id: str) -> bool:
        return any(bad in model_id for bad in self.exclude)


class OpenRouterProvider(Provider):
    name = "openrouter"
    base = "https://openrouter.ai/api/v1"
    key_env = "OPENROUTER_API_KEY"

    def headers(self) -> dict[str, str]:
        return {
            **super().headers(),
            "HTTP-Referer": "http://localhost",
            "X-Title": "counterpoint",
        }

    def payload(self, model, messages, temperature, max_tokens) -> dict:
        body = super().payload(model, messages, temperature, max_tokens)
        # Reasoning models otherwise dump their scratchpad into `content`.
        body["reasoning"] = {"enabled": False}
        return body

    def filter_models(self, data: list[dict]) -> list[str]:
        # Only the free tier is usable without spending credit.
        return [m["id"] for m in data if str(m.get("id", "")).endswith(":free")]


class GroqProvider(Provider):
    name = "groq"
    base = "https://api.groq.com/openai/v1"
    key_env = "GROQ_API_KEY"
    # Speech, moderation, and prompt-injection classifiers: not conversational.
    exclude = ("whisper", "prompt-guard", "orpheus", "safeguard", "tts")

    def payload(self, model, messages, temperature, max_tokens) -> dict:
        body = super().payload(model, messages, temperature, max_tokens)
        # gpt-oss and qwen think out loud; keep the scratchpad out of `content`.
        body["reasoning_format"] = "hidden"
        return body

    def filter_models(self, data: list[dict]) -> list[str]:
        """Zero-priced, text-out, chat-capable models only.

        Groq has no free/paid flag, but it does publish per-token pricing, so
        `prompt == completion == 0` is the honest definition of free. Set
        GROQ_ALLOW_PAID=1 to also list the metered ones.
        """
        allow_paid = os.environ.get("GROQ_ALLOW_PAID", "").strip() in ("1", "true", "yes")
        out = []
        for m in data:
            model_id = str(m.get("id", ""))
            if not model_id or self.is_excluded(model_id) or not m.get("active", True):
                continue
            if "text" not in (m.get("output_modalities") or ["text"]):
                continue
            pricing = m.get("pricing") or {}
            try:
                metered = float(pricing.get("prompt") or 0) > 0 or float(
                    pricing.get("completion") or 0
                ) > 0
            except (TypeError, ValueError):
                metered = True
            if metered and not allow_paid:
                continue
            out.append(model_id)
        return out


class MistralProvider(Provider):
    name = "mistral"
    base = "https://api.mistral.ai/v1"
    key_env = "MISTRAL_API_KEY"
    # Audio, code-completion and CLI-agent endpoints: chat-capable but not
    # conversationalists.
    exclude = ("voxtral", "-fim", "vibe-cli", "ocr", "embed", "moderation")

    def filter_models(self, data: list[dict]) -> list[str]:
        """Chat-capable, non-deprecated models.

        Mistral publishes no per-model pricing, so there is nothing to filter on:
        on the free Experiment tier every model is free but rate-limited, and on a
        paid plan every model is metered. By default only the `-latest` aliases are
        listed, since the catalogue is mostly dated snapshots of the same handful of
        models. Set MISTRAL_ALL_MODELS=1 for the full list.
        """
        all_models = os.environ.get("MISTRAL_ALL_MODELS", "").strip() in ("1", "true", "yes")
        out = []
        for m in data:
            model_id = str(m.get("id", ""))
            caps = m.get("capabilities") or {}
            if not model_id or not caps.get("completion_chat"):
                continue
            if m.get("deprecation") or self.is_excluded(model_id):
                continue
            if not all_models and not model_id.endswith("-latest"):
                continue
            out.append(model_id)
        return out


class CerebrasProvider(Provider):
    name = "cerebras"
    base = "https://api.cerebras.ai/v1"
    key_env = "CEREBRAS_API_KEY"
    # Speech/guard models, should any appear alongside the chat ones.
    exclude = ("whisper", "guard", "embed")

    def payload(self, model, messages, temperature, max_tokens) -> dict:
        body = super().payload(model, messages, temperature, max_tokens)
        # gpt-oss reasons before answering; keep it short so turns stay snappy.
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
        return body

    def filter_models(self, data: list[dict]) -> list[str]:
        """Every listed model.

        Cerebras exposes no pricing or capability metadata, and its free tier
        covers the whole (small) catalogue, so there is nothing to filter beyond
        the non-chat exclusions.
        """
        return [
            m["id"]
            for m in data
            if m.get("id") and not self.is_excluded(str(m["id"]))
        ]


class OllamaCloudProvider(Provider):
    """Ollama Cloud — hosted big models at ollama.com.

    The catalogue is readable anonymously, but inference returns 401 without a
    key, so a key is required for the provider to count as available.
    """

    name = "ollama-cloud"
    base = "https://ollama.com/v1"
    key_env = "OLLAMA_API_KEY"
    exclude = ("embed", "guard")
    # These think in a separate `reasoning` field that is billed against the same
    # max_tokens budget, so a normal debate-sized cap leaves nothing for the
    # actual answer.
    REASONERS = ("gpt-oss", "nemotron", "deepseek", "qwen", "glm", "kimi", "minimax")
    REASONING_FLOOR = 900

    def payload(self, model, messages, temperature, max_tokens) -> dict:
        if any(r in model for r in self.REASONERS):
            max_tokens = max(max_tokens, self.REASONING_FLOOR)
        return super().payload(model, messages, temperature, max_tokens)


class OllamaProvider(Provider):
    """A local Ollama daemon. No key, no quota, no rate limit — just slower.

    Worth keeping at the end of the fallback chain: when every hosted free tier
    is exhausted, the machine under the desk still answers.
    """

    name = "ollama"
    key_env = "OLLAMA_API_KEY"  # optional; local daemons usually need none
    exclude = ("embed", "nomic-embed", "guard")

    @property
    def host(self) -> str:
        return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    @property
    def base(self) -> str:  # type: ignore[override]
        host = self.host
        if not host.startswith("http"):
            host = f"http://{host}"
        return f"{host}/v1"

    @property
    def available(self) -> bool:
        return probe_local_ollama(self.host)

    def why_unavailable(self) -> str:
        return "" if self.available else f"no Ollama daemon reachable at {self.host}"

    def headers(self) -> dict[str, str]:
        head = super().headers()
        if not self.api_key:
            # A local daemon rejects nothing, but an empty bearer is still junk.
            head.pop("Authorization", None)
        return head


_LOCAL_PROBE: dict[str, bool] = {}


def probe_local_ollama(host: str) -> bool:
    """One cheap TCP check per host per process, cached.

    Cached because `available` is consulted repeatedly while building fallback
    chains, and a dead daemon should not cost a connect timeout every time.
    """
    if host in _LOCAL_PROBE:
        return _LOCAL_PROBE[host]
    parsed = urllib.parse.urlparse(host if "//" in host else f"http://{host}")
    port = parsed.port or (443 if parsed.scheme == "https" else 11434)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=0.6):
            ok = True
    except OSError:
        ok = False
    _LOCAL_PROBE[host] = ok
    return ok


class KiloProvider(Provider):
    """Kilo Code — an OpenRouter-shaped gateway on its own quota.

    Same catalogue and response format as OpenRouter, but a separate account and
    separate limits, which is exactly what makes it useful here: the identical
    model can be reached through either vendor when one runs dry.
    """

    name = "kilo"
    base = "https://kilo.ai/api/openrouter"
    key_env = "KILO_API_KEY"
    # Music generation, safety classifiers, and the auto-router pseudo-models —
    # the routers answer with empty content rather than a usable turn.
    exclude = (
        "lyria",
        "content-safety",
        "guard",
        "embed",
        "moderation",
        "kilo-auto",
        "openrouter/",
        "stealth/",
    )

    def payload(self, model, messages, temperature, max_tokens) -> dict:
        body = super().payload(model, messages, temperature, max_tokens)
        # Proxies OpenRouter, so it honours the same reasoning switch.
        body["reasoning"] = {"enabled": False}
        return body

    def filter_models(self, data: list[dict]) -> list[str]:
        """Zero-priced models only. Kilo publishes OpenRouter's pricing block."""
        out = []
        for m in data:
            model_id = str(m.get("id", ""))
            if not model_id or self.is_excluded(model_id):
                continue
            pricing = m.get("pricing") or {}
            try:
                metered = float(pricing.get("prompt") or 0) > 0 or float(
                    pricing.get("completion") or 0
                ) > 0
            except (TypeError, ValueError):
                metered = True
            if metered:
                continue
            out.append(model_id)
        return out


class CloudflareProvider(Provider):
    """Workers AI.

    The odd one out: its REST paths are scoped to an account, so the base URL is
    built per call and an account id is required alongside the token. The
    OpenAI-compatible chat endpoint lives under `/ai/v1`, but the model catalogue
    does not — that is Cloudflare's own `/ai/models/search`, so `list_models` is
    overridden rather than inheriting the `/models` convention.
    """

    name = "cloudflare"
    key_env = "CLOUDFLARE_API_TOKEN"
    account_env = "CLOUDFLARE_ACCOUNT_ID"
    exclude = ("guard", "embed", "whisper", "bge-", "rerank")

    @property
    def account_id(self) -> str:
        return os.environ.get(self.account_env, "").strip()

    @property
    def base(self) -> str:  # type: ignore[override]
        return (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/v1"
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.account_id)

    def why_unavailable(self) -> str:
        if not self.api_key:
            return f"no API token set ({self.key_env})"
        if not self.account_id:
            return f"no account id set ({self.account_env})"
        return ""

    def list_models(self, timeout: int = 30) -> list[str]:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            "/ai/models/search?task=Text%20Generation&per_page=100"
        )
        req = urllib.request.Request(url, headers=self.headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("success", True):
            errors = "; ".join(e.get("message", "") for e in body.get("errors") or [])
            raise LLMError(f"cloudflare model list: {errors or 'request failed'}")
        return sorted(self.filter_models(body.get("result") or []))

    def filter_models(self, data: list[dict]) -> list[str]:
        """Text-generation models, minus embedding/guard/audio entries.

        Workers AI bills every model out of one shared daily neuron allowance
        rather than pricing them individually, so there is no per-model free flag
        to filter on — the free tier is the account's daily quota.
        """
        out = []
        for m in data:
            model_id = str(m.get("name") or "")
            task = ((m.get("task") or {}).get("name") or "").lower()
            if not model_id or self.is_excluded(model_id):
                continue
            if task and "text generation" not in task:
                continue
            out.append(model_id)
        return out


PROVIDERS: dict[str, Provider] = {}

# model id -> params it 400'd on, so later calls omit them up front.
_REJECTED: dict[str, set[str]] = {}

# 429s that mean "come back tomorrow" rather than "wait a few seconds".
_QUOTA_MARKERS = (
    "per-day",
    "per day",
    "daily",
    "quota",
    "add 10 credits",
    "insufficient",
    "monthly",
)


def is_quota_exhausted(detail: str) -> bool:
    low = detail.lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


# A 401/403 usually means the whole account is unusable — but not when the
# provider is really saying "that particular model needs a better plan". Those
# must not disable the sibling models that do work on this account.
_MODEL_SCOPED_MARKERS = (
    "this model",
    "model requires",
    "requires a subscription",
    "upgrade",
    "not available on your plan",
    "no access to model",
)


def is_model_scoped(detail: str) -> bool:
    low = detail.lower()
    return any(marker in low for marker in _MODEL_SCOPED_MARKERS)


# provider name -> why it is unusable, set on billing/auth failures so the rest of
# the run skips it instead of paying a round trip per turn to be told again.
_DEAD_PROVIDERS: dict[str, str] = {}


def split_model(model_id: str) -> tuple[Provider, str]:
    """'groq:openai/gpt-oss-120b' -> (GroqProvider, 'openai/gpt-oss-120b')."""
    prefix, sep, rest = model_id.partition(":")
    if sep and prefix in PROVIDERS:
        return PROVIDERS[prefix], rest
    return PROVIDERS[DEFAULT_PROVIDER], model_id


def qualify(provider_name: str, model: str) -> str:
    return f"{provider_name}:{model}"


class LLMClient:
    """Routes a namespaced model id to the right provider, with retries."""

    def __init__(self, *, timeout: int = 120, max_retries: int = 4) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        # Callers (CLI, web server) swap this to route progress chatter.
        self.on_notice = lambda msg: print(f"    [{msg}]")
        if not any(p.available for p in PROVIDERS.values()):
            raise LLMError(
                "No API keys. Set OPENROUTER_API_KEY and/or GROQ_API_KEY in .env"
            )

    def chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        temperature: float = 0.8,
        max_tokens: int = 700,
    ) -> str:
        provider, model = split_model(model_id)
        if not provider.available:
            raise LLMError(f"{provider.name}: {provider.why_unavailable()}")
        if provider.name in _DEAD_PROVIDERS:
            raise LLMError(f"{provider.name} unusable: {_DEAD_PROVIDERS[provider.name]}")

        body = provider.payload(model, messages, temperature, max_tokens)
        # Skip params this model already rejected once, instead of re-learning
        # it on every single turn.
        for param in _REJECTED.get(model_id, ()):
            body.pop(param, None)
        payload = json.dumps(body).encode("utf-8")
        url = f"{provider.base}/chat/completions"

        delay = 2.0
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url, data=payload, headers=provider.headers(), method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
                # A provider that rejects one of our optional params: drop it, retry.
                if exc.code == 400:
                    dropped = self._drop_rejected_param(body, detail)
                    if dropped:
                        _REJECTED.setdefault(model_id, set()).add(dropped)
                        self.on_notice(f"{model} rejected `{dropped}`, retrying without")
                        payload = json.dumps(body).encode("utf-8")
                        continue
                # "That model needs a paid plan" is about one model, not the key —
                # fail just this model so its working siblings stay usable.
                if exc.code in (401, 402, 403) and is_model_scoped(detail):
                    raise LLMError(
                        f"{provider.name}/{model} not on this plan: {detail[:140]}"
                    ) from exc
                # Billing or credential problems are account-level: every model
                # behind this key will fail the same way for the whole run.
                if exc.code in (401, 402, 403):
                    reason = {
                        401: "bad API key, or wrong account/scope",
                        402: "payment required / no quota on this account",
                        403: "access forbidden",
                    }[exc.code]
                    _DEAD_PROVIDERS[provider.name] = reason
                    self.on_notice(f"{provider.name} disabled for this run: {reason}")
                    raise LLMError(f"{provider.name}: {reason} — {detail[:120]}") from exc
                # A daily/monthly quota will not clear during a run: burning the
                # backoff ladder on it just stalls the debate. Fail now so the
                # caller can switch to another provider immediately.
                if exc.code == 429 and is_quota_exhausted(detail):
                    raise LLMError(f"{provider.name} quota exhausted: {detail[:160]}") from exc
                # 429 = rate limited, 5xx = upstream hiccup.
                if exc.code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    self.on_notice(f"retry {attempt}/{self.max_retries}: {last_error[:110]}")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = f"network error: {exc.reason}"
                if attempt < self.max_retries:
                    self.on_notice(f"retry {attempt}/{self.max_retries}: {last_error}")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(last_error) from exc

            if "error" in data and not data.get("choices"):
                raise LLMError(f"API error: {data['error']}")
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(f"Empty response: {json.dumps(data)[:300]}")

            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            if text:
                return text

            # Empty content with a populated `reasoning` field means the model
            # spent the whole budget thinking and never got to the answer.
            # Returning the scratchpad would put raw chain-of-thought into the
            # debate, so fail and let the caller retry or switch models.
            if (message.get("reasoning") or "").strip():
                raise LLMError(
                    f"{model}: reasoning-only response, answer budget exhausted"
                )
            raise LLMError(f"Blank content: {json.dumps(data)[:300]}")

        raise LLMError(last_error or "unknown failure")

    @staticmethod
    def _drop_rejected_param(body: dict, detail: str) -> str | None:
        """Remove whichever optional param the provider complained about.

        Matched case-insensitively: some endpoints answer "Reasoning is mandatory
        for this endpoint and cannot be disabled", which is the same signal as a
        lowercase param name — drop our override and let the model do its thing.
        """
        low = detail.lower()
        for param in ("reasoning_format", "reasoning_effort", "reasoning"):
            if param in low and param in body:
                body.pop(param)
                return param
        return None

    def models_by_provider(self) -> dict[str, list[str]]:
        """Namespaced model ids per provider. Unreachable providers are skipped."""
        out: dict[str, list[str]] = {}
        for name, provider in PROVIDERS.items():
            if not provider.available:
                continue
            try:
                out[name] = [qualify(name, m) for m in provider.list_models(self.timeout)]
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                self.on_notice(f"{name} model list unavailable: {exc}")
        return out

    def all_models(self) -> list[str]:
        return [m for models in self.models_by_provider().values() for m in models]


for _p in (
    OpenRouterProvider(),
    GroqProvider(),
    MistralProvider(),
    CerebrasProvider(),
    CloudflareProvider(),
    KiloProvider(),
    OllamaCloudProvider(),
    OllamaProvider(),
):
    PROVIDERS[_p.name] = _p


# Back-compat shim for the original single-provider entry point.
class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        kwargs.pop("referer", None)
        kwargs.pop("title", None)
        super().__init__(**kwargs)

    def list_free_models(self) -> list[str]:
        return self.all_models()
