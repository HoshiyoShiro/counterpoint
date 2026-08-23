"""Counterpoint — two AI agents discuss a topic with each other and form an opinion.

Usage:
    python debate.py "Should small teams use microservices?"
    python debate.py "topic" --rounds 5 --mode debate
    python debate.py --list-models
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

from providers import LLMClient, LLMError, load_env, split_model

# Older name, kept so existing callers/except-blocks still work.
OpenRouterError = LLMError

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "transcripts"

MODEL_A = "mistral:mistral-small-latest"
MODEL_B = "groq:groq/compound-mini"
# Third model acts as the neutral judge writing the synthesis, so no debater
# grades its own argument.
MODEL_JUDGE = "mistral:mistral-medium-latest"

# Free tiers 429 constantly, so the chain alternates vendors: each provider's
# rate limit is independent, and a jump to a different vendor is far likelier to
# succeed than another wait on the same saturated pool.
FALLBACKS = [
    "mistral:mistral-small-latest",
    "groq:groq/compound-mini",
    "mistral:ministral-8b-latest",
    "groq:groq/compound",
    "mistral:mistral-medium-latest",
    "mistral:ministral-14b-latest",
    "cerebras:gpt-oss-120b",
    "cerebras:gemma-4-31b",
    "cloudflare:@cf/meta/llama-3.1-8b-instruct",
    # Kilo reaches the same catalogue as OpenRouter but on a separate quota, so
    # it is worth trying before OpenRouter's own daily cap.
    "kilo:nvidia/nemotron-3-super-120b-a12b:free",
    "kilo:thinkingmachines/inkling:free",
    # Verified reachable on a keyed free Ollama account; the rest of the cloud
    # catalogue is subscription-gated and 403s per model.
    "ollama-cloud:gemma4:31b",
    "ollama-cloud:minimax-m3",
    "ollama-cloud:gpt-oss:120b",
    # OpenRouter's free tier has a hard daily request cap, so it sits last: it is
    # the first thing to run dry and the only one that stays dry until reset.
    "openrouter:nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter:google/gemma-4-31b-it:free",
    "openrouter:z-ai/glm-5.2:free",
    "openrouter:nvidia/nemotron-nano-9b-v2:free",
    # Absolute last resort: a local daemon has no quota to exhaust. Slower, and
    # only present when one is actually running.
    "ollama:llama3.2:latest",
]

CONVERGE_TAG = "[CONVERGED]"
# max_tokens can cut the tag off mid-word ("[CONVERGE", "[CONVERGED"), which would
# both hide the signal and leave a broken fragment in the transcript.
CONVERGE_RE = re.compile(r"\[\s*CONVERGE[D]?\s*\]?\s*$", re.I)


def took_the_tag(reply: str) -> tuple[bool, str]:
    """(did the model signal done, reply with the tag removed)."""
    signalled = CONVERGE_TAG in reply or bool(CONVERGE_RE.search(reply))
    clean = CONVERGE_RE.sub("", reply.replace(CONVERGE_TAG, "")).strip()
    return signalled, clean


def is_configured(model_id: str) -> bool:
    """True when the provider behind this model id has its credentials set."""
    provider, _ = split_model(model_id)
    return provider.available


# Persona pairs per discussion mode.
MODES = {
    "discuss": (
        (
            "Ada",
            "You think in systems and evidence. You start from first principles, "
            "quantify trade-offs, and are quick to say when the other agent has "
            "changed your mind.",
        ),
        (
            "Rune",
            "You think in second-order effects and human factors. You probe "
            "assumptions, raise failure modes the other agent skipped, and you "
            "concede a point cleanly when it is well argued.",
        ),
    ),
    "debate": (
        (
            "Pro",
            "You argue FOR the proposition. Make the strongest honest case, attack "
            "weak reasoning, but never misrepresent evidence or the other side.",
        ),
        (
            "Con",
            "You argue AGAINST the proposition. Make the strongest honest case, "
            "attack weak reasoning, but never misrepresent evidence or the other side.",
        ),
    ),
    "review": (
        (
            "Architect",
            "You propose concrete designs and defend them with reasoning about "
            "constraints, cost, and maintenance burden.",
        ),
        (
            "Critic",
            "You stress-test proposals: edge cases, operational cost, simpler "
            "alternatives. You approve a design once it genuinely holds up.",
        ),
    ),
}

MAX_COUNCIL_ROUNDS = 20

# Distinct thinking styles, so a council of eight does not produce eight of the
# same answer. Seats are filled from the top of this list.
COUNCIL_PERSONAS = [
    ("Ada", "You think in systems and evidence, and you quantify trade-offs."),
    ("Rune", "You think in second-order effects and human factors."),
    ("Vex", "You are adversarial: hunt the failure mode everyone else skipped."),
    ("Juno", "You are a pragmatist: ask what actually ships this week."),
    ("Pike", "You are the historian: what was tried before, and how did it go?"),
    ("Wren", "You are the simplifier: argue for the least machinery that works."),
    ("Cato", "You weigh risk and cost: who pays when this breaks?"),
    ("Nell", "You are the user's advocate: how does this feel from outside?"),
    ("Iris", "You are the synthesist: find the shared frame between positions."),
    ("Otto", "You are the empiricist: demand a measurement for every claim."),
    ("Sable", "You watch for scope creep and unstated assumptions."),
    ("Bram", "You argue from operational reality: on-call, migrations, rollback."),
]

# Which model to seat from each provider, best first. Anything not listed falls
# back to whatever that provider lists first.
COUNCIL_PICKS = {
    "mistral": ["mistral-small-latest", "ministral-8b-latest", "mistral-medium-latest"],
    "groq": ["groq/compound-mini", "groq/compound"],
    "cerebras": ["gemma-4-31b", "gpt-oss-120b"],
    "kilo": ["tencent/hy3:free", "thinkingmachines/inkling:free",
             "nvidia/nemotron-3-super-120b-a12b:free"],
    # Only the models confirmed reachable without an Ollama subscription.
    "ollama-cloud": ["gemma4:31b", "minimax-m3", "nemotron-3-super"],
    "ollama": ["llama3.2:latest", "qwen3:8b"],
    "openrouter": ["nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-31b-it:free"],
    "cloudflare": ["@cf/meta/llama-3.1-8b-instruct"],
}

SHARED_RULES = """You are in a live conversation with another AI agent. This is a real
exchange, not a transcript you narrate.

Rules:
- Address the other agent directly and respond to what they actually said.
- Be concrete. Prefer specifics, numbers, and named mechanisms over abstractions.
- Max ~150 words per turn. No headers, no bullet-point walls of text.
- Do not restate their point back to them before replying. Just reply.
- Change your position when the argument warrants it, and say so explicitly.
- When you and the other agent have genuinely reached a shared position and further
  turns would add nothing, end your message with the exact tag {tag}.
- Never write the other agent's lines for them.
"""

SYNTHESIS_PROMPT = """Below is a discussion between two AI agents on the topic: "{topic}"

{transcript}

Write the joint conclusion, in this exact structure:

## Position
The opinion they converged on, in 2-4 sentences. If they did not converge, say so and
state each position.

## Agreed
- Points both accepted.

## Unresolved
- Genuine remaining disagreements, or "None" if there are none.

## Confidence
One line: high / medium / low, and why.

Report only what the discussion supports. Do not add new arguments of your own.
Output the four sections and nothing else. Do not show your reasoning, do not
narrate your process, and start your reply with the literal text "## Position".
"""

ALL_MODES = ["discuss", "debate", "review", "council"]

GOALS = {
    "discuss": "Work with the other agent toward a well-reasoned shared opinion.",
    "debate": "Argue your assigned side, but follow the evidence if it turns.",
    "review": "Reach a design you would both sign off on.",
}

COUNCIL_RULES = """You are one member of a council of {count} AI models, each from a
different vendor, discussing this in one shared room. Every other member sees what
you say.

Members: {roster}

Rules:
- Speak as {name}, in your own voice. Reply to what others actually said, by name.
- Do not repeat a point someone already made. Add, sharpen, or challenge.
- Max ~120 words. No headers, no bullet lists.
- Say plainly when someone changes your mind.
- Do not summarise the discussion. A separate judge does that at the end.
- Never write another member's lines for them.
- The council decides for itself when it is finished. When you believe the group has
  said what it needs to and further rounds would only repeat, end your message with
  the exact tag {tag}. The discussion ends only once every member does this in the
  same round, so do not use it merely because you personally have nothing to add —
  use it when the *group* is done.
"""


def build_council_prompt(name: str, persona: str, roster: list[str], topic: str,
                         count: int) -> str:
    others = ", ".join(roster)
    return (
        f"You are {name}. {persona}\n\n"
        f"Topic: {topic}\n\n"
        + COUNCIL_RULES.format(
            count=count, roster=others, name=name, tag=CONVERGE_TAG
        )
    )


def build_bench(client: LLMClient, exclude: set[str] | None = None) -> list[str]:
    """Every usable model, interleaved across providers.

    Used as the substitute pool: taking the next entry gives you a model from a
    *different* vendor than the one that just failed, which is the whole reason a
    replacement is likely to work at all.
    """
    exclude = exclude or set()
    by_provider = []
    for provider_name, models in client.models_by_provider().items():
        usable = [m for m in models if m not in exclude]
        if not usable:
            continue
        # Hand-picked conversational models first; the rest of the catalogue
        # (code-completion, tiny, or unproven models) only after those run out.
        preferred = [
            f"{provider_name}:{want}"
            for want in COUNCIL_PICKS.get(provider_name, [])
            if f"{provider_name}:{want}" in usable
        ]
        by_provider.append(preferred + [m for m in usable if m not in preferred])
    bench: list[str] = []
    for column in range(max((len(c) for c in by_provider), default=0)):
        for models in by_provider:
            if column < len(models):
                bench.append(models[column])
    return bench


def substitute(model_id: str, bench: list[str], taken: set[str]) -> str | None:
    """First bench model that is configured, unused, and from another provider."""
    dead_provider, _ = split_model(model_id)
    same_provider_spare = None
    for candidate in bench:
        if candidate in taken or candidate == model_id:
            continue
        provider, _ = split_model(candidate)
        if not provider.available:
            continue
        if provider.name != dead_provider.name:
            return candidate
        same_provider_spare = same_provider_spare or candidate
    return same_provider_spare


def build_council(client: LLMClient, per_provider: int = 1) -> list[tuple[str, str]]:
    """Seat 1-2 models from each configured provider.

    Returns [(display_name, namespaced_model_id)]. Providers that are configured
    but list nothing are skipped; a provider whose preferred picks are all absent
    contributes whatever it does list.
    """
    per_provider = max(1, min(2, per_provider))
    seats: list[str] = []

    for provider_name, models in client.models_by_provider().items():
        if not models:
            continue
        bare = {m.split(":", 1)[1]: m for m in models}
        chosen: list[str] = []
        for want in COUNCIL_PICKS.get(provider_name, []):
            if want in bare and bare[want] not in chosen:
                chosen.append(bare[want])
            if len(chosen) == per_provider:
                break
        # Nothing preferred available: take the provider's first listings.
        for m in models:
            if len(chosen) >= per_provider:
                break
            if m not in chosen:
                chosen.append(m)
        seats.extend(chosen[:per_provider])

    # One persona per seat, never reused: two members answering to "Ada" would
    # make every reply ambiguous about who is being addressed.
    seats = seats[: len(COUNCIL_PERSONAS)]
    return [(COUNCIL_PERSONAS[i][0], model) for i, model in enumerate(seats)]


def build_system_prompt(name: str, persona: str, other: str, topic: str, mode: str) -> str:
    return (
        f"You are {name}. {persona}\n\n"
        f"You are talking with {other}.\n"
        f"Topic: {topic}\n"
        f"Goal: {GOALS[mode]}\n\n" + SHARED_RULES.format(tag=CONVERGE_TAG)
    )


THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
OPEN_THINK = re.compile(r"^\s*<(think|thinking|reasoning)>.*", re.S | re.I)
# Groq's compound models answer as "**Reasoning** … **Answer** …" — keep the answer.
MD_REASONING = re.compile(
    r"^\s*\*\*(?:reasoning|thought|analysis)\*\*.*?\*\*(?:answer|response|reply|final)\*\*:?\s*",
    re.S | re.I,
)
# …and sometimes there is no answer header, just the header and the prose.
MD_REASONING_HEAD = re.compile(r"^\s*\*\*(?:reasoning|thought|analysis)\*\*:?\s*", re.I)
# Some free reasoning models narrate before answering; cut the preamble.
PREAMBLE = re.compile(
    r"^\s*(?:okay|alright|let me|i need to|i should|i must|first,|the user)\b.*?\n\n",
    re.S | re.I,
)


def strip_reasoning(text: str, speakers: list[str]) -> str:
    """Remove leaked chain-of-thought and speaker-name prefixes from a reply."""
    text = THINK_BLOCK.sub("", text)
    text = OPEN_THINK.sub("", text)
    if MD_REASONING.search(text):
        text = MD_REASONING.sub("", text)
    else:
        text = MD_REASONING_HEAD.sub("", text)
    text = PREAMBLE.sub("", text)
    # Models often prefix their own name or the addressee's: "Ada: Rune, ..."
    for _ in range(2):
        for name in speakers:
            text = re.sub(rf"^\s*\**{re.escape(name)}\**\s*:\s*", "", text)
    return text.strip()


class Agent:
    """One participant. Keeps its own view of the conversation."""

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        client: LLMClient,
        speakers: list[str],
    ):
        self.name = name
        self.model = model
        self.client = client
        self.speakers = speakers
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def hear(self, speaker: str, text: str) -> None:
        self.history.append({"role": "user", "content": f"{speaker}: {text}"})

    def speak(
        self,
        temperature: float,
        max_tokens: int,
        fallbacks: list[str],
        taken: frozenset[str] = frozenset(),
    ) -> str:
        # `taken` holds models other seats are already using: a replacement should
        # bring a new voice to the room, not clone one that is already speaking.
        candidates = [self.model] + [
            m for m in fallbacks if m != self.model and m not in taken
        ]
        last_exc: LLMError | None = None

        for model in candidates:
            for attempt in (1, 2):
                try:
                    raw = self.client.chat(
                        model, self.history, temperature=temperature, max_tokens=max_tokens
                    )
                except LLMError as exc:
                    last_exc = exc
                    self.client.on_notice(f"model down: {model} — {str(exc)[:100]}")
                    break

                reply = strip_reasoning(raw, self.speakers)
                # A stub like "Ada" means the model addressed and stopped; retry once.
                if len(reply) < 40:
                    if attempt == 1:
                        self.client.on_notice(f"degenerate reply from {model}, retrying")
                        continue
                    last_exc = LLMError(f"{model} kept returning stub replies")
                    break

                if model != self.model:
                    self.client.on_notice(f"switched {self.name}: {self.model} -> {model}")
                    self.model = model
                self.history.append({"role": "assistant", "content": reply})
                return reply

        raise last_exc or LLMError("no model available")


def wrap(text: str, indent: str = "    ") -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.append(
                textwrap.fill(para, width=96, initial_indent=indent, subsequent_indent=indent)
            )
        else:
            lines.append("")
    return "\n".join(lines)


def converse(args: argparse.Namespace) -> Iterator[dict]:
    """Run a full discussion, yielding events as they happen.

    Event types: meta, turn-start, turn, notice, converged, synthesis, saved, error.
    Both the CLI and the web server consume this same stream.
    """
    client = LLMClient()
    notices: list[str] = []
    client.on_notice = notices.append  # drained after each blocking call

    # Drop fallbacks whose provider has no key configured — otherwise every turn
    # pays a guaranteed failure and prints a notice about it.
    fallbacks = [] if args.no_fallback else [m for m in FALLBACKS if is_configured(m)]

    bench = build_bench(client)
    for note in notices:
        yield {"type": "notice", "text": note}
    notices.clear()

    # Pre-flight: a model whose provider has no credentials would fail on its
    # very first turn, so swap it now and say so, rather than opening with an
    # error the user has to read past.
    if args.mode != "council":
        chosen: set[str] = set()
        for slot in ("model_a", "model_b", "model_judge"):
            picked = getattr(args, slot)
            if is_configured(picked):
                chosen.add(picked)
                continue
            stand_in = substitute(picked, bench, chosen)
            if not stand_in:
                yield {"type": "error", "text": f"No configured provider can fill {slot}."}
                return
            yield {
                "type": "notice",
                "text": f"{picked} unavailable — seating {stand_in} instead",
            }
            setattr(args, slot, stand_in)
            chosen.add(stand_in)

    council = args.mode == "council"
    if council:
        seats = build_council(client, getattr(args, "per_provider", 1))
        if len(seats) < 2:
            yield {
                "type": "error",
                "text": "Council needs at least two providers configured.",
            }
            return
        roster = [name for name, _ in seats]
        personas = dict(COUNCIL_PERSONAS)
        agents = [
            Agent(
                name,
                model,
                build_council_prompt(
                    name, personas[name], [n for n in roster if n != name],
                    args.topic, len(seats),
                ),
                client,
                roster + ["Moderator"],
            )
            for name, model in seats
        ]
        # The council decides when it is done; the cap is only a backstop.
        total_rounds = min(args.rounds or MAX_COUNCIL_ROUNDS, MAX_COUNCIL_ROUNDS)
        # A council seat that dies is replaced from the whole bench, not just the
        # hand-tuned FALLBACKS list, so the room keeps its vendor spread.
        fallbacks = fallbacks or [] if args.no_fallback else bench
    else:
        (name_a, persona_a), (name_b, persona_b) = MODES[args.mode]
        if args.names:
            name_a, name_b = [n.strip() for n in args.names.split(",", 1)]
        roster = [name_a, name_b]
        speakers = roster + ["Moderator"]
        agents = [
            Agent(
                name_a,
                args.model_a,
                build_system_prompt(name_a, persona_a, name_b, args.topic, args.mode),
                client,
                speakers,
            ),
            Agent(
                name_b,
                args.model_b,
                build_system_prompt(name_b, persona_b, name_a, args.topic, args.mode),
                client,
                speakers,
            ),
        ]
        total_rounds = args.rounds

    yield {
        "type": "meta",
        "topic": args.topic,
        "mode": args.mode,
        "rounds": total_rounds,
        "capped": MAX_COUNCIL_ROUNDS if council else None,
        "agents": [{"name": a.name, "model": a.model} for a in agents],
        "judge": args.model_judge,
    }

    transcript: list[dict] = []
    agents[0].hear("Moderator", f"Open the discussion on: {args.topic}")

    aborted = False
    for round_no in range(1, total_rounds + 1):
        agreed: set[str] = set()

        for speaker in list(agents):
            if speaker not in agents:  # dropped mid-round after a failure
                continue
            yield {
                "type": "turn-start",
                "round": round_no,
                "speaker": speaker.name,
                "model": speaker.model,
            }
            try:
                others = frozenset(a.model for a in agents if a is not speaker)
                reply = speaker.speak(
                    args.temperature, args.max_tokens, fallbacks, others
                )
            except LLMError as exc:
                for note in notices:
                    yield {"type": "notice", "text": note}
                notices.clear()
                hint = ""
                if "429" in str(exc):
                    hint = " Free-tier rate limit — wait a minute or pick other models."
                yield {"type": "error", "text": f"{speaker.name} failed: {exc}{hint}"}
                # One dead seat should not end a whole council; drop it and go on.
                if len(agents) > 2:
                    agents.remove(speaker)
                    yield {
                        "type": "notice",
                        "text": f"{speaker.name} left the council ({len(agents)} remain)",
                    }
                    continue
                aborted = True
                break
            for note in notices:
                yield {"type": "notice", "text": note}
            notices.clear()

            signalled, clean = took_the_tag(reply)
            entry = {
                "round": round_no,
                "speaker": speaker.name,
                "model": speaker.model,
                "text": clean,
            }
            transcript.append(entry)
            yield {"type": "turn", **entry}

            if signalled:
                agreed.add(speaker.name)

            for listener in agents:
                if listener is not speaker:
                    listener.hear(speaker.name, clean)

        if aborted:
            break
        # The discussion ends only when every seat still in the room signalled
        # done within the same round — one member going quiet is not consensus.
        if round_no >= args.min_rounds and agreed >= {a.name for a in agents}:
            yield {"type": "converged", "round": round_no}
            break

    if not transcript:
        yield {"type": "error", "text": "No turns completed."}
        return

    flat = "\n\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)
    judge_prompt = [
        {"role": "user", "content": SYNTHESIS_PROMPT.format(topic=args.topic, transcript=flat)}
    ]
    synthesis = ""
    judge_used = args.model_judge
    for model in [args.model_judge] + [m for m in fallbacks if m != args.model_judge]:
        try:
            raw = client.chat(model, judge_prompt, temperature=0.3, max_tokens=800)
        except LLMError as exc:
            notices.clear()
            yield {"type": "notice", "text": f"judge down: {model} — {str(exc)[:100]}"}
            continue
        notices.clear()
        raw = strip_reasoning(raw, [t["speaker"] for t in transcript])
        # Drop any narration that survived before the first required heading.
        head = raw.find("## Position")
        synthesis = raw[head:] if head > 0 else raw
        judge_used = model
        break
    if not synthesis:
        synthesis = "(synthesis failed: every judge model was rate-limited)"
    yield {"type": "synthesis", "judge": judge_used, "text": synthesis}

    lineup = [(a.name, a.model) for a in agents]
    md, js = save_transcript(args, transcript, synthesis, judge_used, lineup)
    yield {"type": "saved", "md": str(md), "json": str(js), "name": js.stem}


def save_transcript(
    args: argparse.Namespace,
    transcript: list[dict],
    synthesis: str,
    judge_used: str,
    lineup: list[tuple[str, str]],
) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower())[:50].strip("-") or "topic"
    TRANSCRIPTS.mkdir(exist_ok=True)

    body = "\n\n".join(
        f"**{t['speaker']}** (round {t['round']}):\n\n{t['text']}" for t in transcript
    )
    md = TRANSCRIPTS / f"{stamp}-{slug}.md"
    md.write_text(
        f"# {args.topic}\n\n"
        f"- Mode: {args.mode}\n"
        + "".join(f"- {name}: `{model}`\n" for name, model in lineup)
        + f"- Judge: `{judge_used}`\n"
        f"- Date: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        f"## Discussion\n\n{body}\n\n## Synthesis\n\n{synthesis}\n",
        encoding="utf-8",
    )
    js = TRANSCRIPTS / f"{stamp}-{slug}.json"
    js.write_text(
        json.dumps(
            {
                "topic": args.topic,
                "mode": args.mode,
                "date": dt.datetime.now().isoformat(timespec="seconds"),
                "judge": judge_used,
                "lineup": [{"name": n, "model": m} for n, m in lineup],
                "turns": transcript,
                "synthesis": synthesis,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md, js


def run_debate(args: argparse.Namespace) -> int:
    """CLI renderer over the converse() event stream."""
    failed = False
    for ev in converse(args):
        kind = ev["type"]
        if kind == "meta":
            print(f"\nTopic:  {ev['topic']}")
            print(f"Mode:   {ev['mode']}   Rounds: {ev['rounds']}")
            for a in ev["agents"]:
                print(f"{a['name']:<10} {a['model']}")
            print("=" * 100)
        elif kind == "turn-start":
            print(f"\n[round {ev['round']}] {ev['speaker']} ({ev['model']})")
        elif kind == "notice":
            print(f"    [{ev['text']}]")
        elif kind == "turn":
            print(wrap(ev["text"]))
        elif kind == "converged":
            print(f"\n-- both agents converged after round {ev['round']} --")
        elif kind == "synthesis":
            print("\n" + "=" * 100)
            print("SYNTHESIS")
            print("=" * 100)
            print(f"judge: {ev['judge']}\n")
            print(wrap(ev["text"], indent=""))
        elif kind == "saved":
            print(f"\nSaved: {ev['md']}\n       {ev['json']}")
        elif kind == "error":
            print(f"\n!! {ev['text']}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def main() -> int:
    # Windows consoles default to cp1252 and blow up on model output containing
    # en-dashes, non-breaking hyphens, emoji, etc.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    load_env(HERE / ".env")

    p = argparse.ArgumentParser(description="Two AI agents discuss a topic.")
    p.add_argument("topic", nargs="?", help="what the agents should discuss")
    p.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="rounds; 0 means let them decide, capped at --max-rounds (default 4 for "
             "two-agent modes)",
    )
    p.add_argument(
        "--min-rounds",
        type=int,
        default=2,
        help="ignore convergence before this round, so they cannot agree instantly",
    )
    p.add_argument("--mode", choices=ALL_MODES, default="discuss")
    p.add_argument(
        "--council",
        action="store_true",
        help=f"seat 1-2 models from every configured provider (cap {MAX_COUNCIL_ROUNDS} rounds)",
    )
    p.add_argument(
        "--per-provider",
        type=int,
        default=1,
        choices=(1, 2),
        help="council seats per provider (default 1)",
    )
    p.add_argument("--model-a", default=MODEL_A)
    p.add_argument("--model-b", default=MODEL_B)
    p.add_argument("--model-judge", default=MODEL_JUDGE, help="neutral synthesis model")
    p.add_argument("--names", help="override agent names, e.g. 'Alice,Bob'")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail instead of switching models when one is rate-limited",
    )
    p.add_argument(
        "--list-models", action="store_true", help="list free models across all providers"
    )
    args = p.parse_args()

    if args.list_models:
        for provider, models in LLMClient().models_by_provider().items():
            print(f"\n{provider}:")
            for m in models:
                print(f"  {m}")
        return 0

    if args.council:
        args.mode = "council"
    if args.rounds <= 0:
        args.rounds = MAX_COUNCIL_ROUNDS if args.mode == "council" else 4
    args.rounds = min(args.rounds, MAX_COUNCIL_ROUNDS)

    if not args.topic:
        p.error("topic required (or use --list-models)")

    try:
        return run_debate(args)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
