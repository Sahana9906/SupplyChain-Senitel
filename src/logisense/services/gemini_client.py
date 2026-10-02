"""LLM clients for the Phase 1 Triage Agent and Phase 2 Hypothesis Agent.

NOTE ON NAMING: this module, its classes (GeminiIntakeClient /
GeminiHypothesisClient) and the shared build_gemini_client() helper keep
their original "Gemini" names on purpose -- the project migrated its
underlying LLM provider from Google Gemini to Groq, but per the explicit
migration requirement to make the smallest possible change and not rename
business classes unnecessarily, only the *implementation* changed. Every
call site elsewhere in the project (api.py, cli.py, evidence_validation.py,
multi_source_investigation.py, rag/hybrid_investigation.py,
rag/qa_service.py, rag/embeddings.py) is completely unaffected by the
provider swap: `build_gemini_client(settings)` still returns an object
shaped exactly like `google.genai.Client` (a `.models.generate_content(
model=, contents=, config=)` method returning `.text` / `.usage_metadata`),
so no other file needed to change its call sites, and every existing test
fake (FakeGeminiClient / FakeGeminiModels, which mock this exact shape)
keeps working unchanged.

Phase 1 (Triage Agent) returns human-readable text with a fixed set of
section headings (not JSON -- this text is meant to be shown to an
operations user directly). Phase 2 (Hypothesis Agent) returns a
structured JSON list, since hypotheses are inherently sub-structured list
items rather than a single narrative -- a deliberate per-phase choice.

Both clients parse/validate their model's output and retry exactly once
with a correction prompt on failure. If it fails again, the caller must
mark the investigation FAILED -- neither client ever fabricates a result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import groq
from pydantic import BaseModel, Field, ValidationError, field_validator

from logisense.config import Settings
from logisense.services.prompts import HypothesisPrompt, PhaseOnePrompt

logger = logging.getLogger(__name__)

# Groq's Python SDK (like the OpenAI SDK it's forked from) already retries
# 408/409/429/5xx responses with exponential backoff + jitter internally
# (see groq._base_client.SyncAPIClient._should_retry /
# _calculate_retry_timeout) -- so, exactly as with the previous Gemini
# retry configuration, we only need to raise its `max_retries` count
# rather than writing a second, duplicate retry loop on top of it. Groq
# logs nothing on its own for retries, so we log the final outcome
# ourselves in each _call() below instead.
_RETRYABLE_STATUS_CODES = (408, 409, 429, 500, 502, 503, 504)


# ---------------------------------------------------------------------------
# Groq adapter -- gives the Groq SDK the exact same call shape the rest of
# this project already uses (`client.models.generate_content(model=,
# contents=, config=)` -> object with `.text` / `.usage_metadata`), which
# is the shape every call site AND every existing test fake was written
# against. This is the only place raw Groq SDK calls happen.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _UsageMetadata:
    total_token_count: int | None


@dataclass(frozen=True)
class _GenerateContentResult:
    text: str | None
    usage_metadata: _UsageMetadata


class _GroqModelsAdapter:
    """Mimics `genai.Client(...).models` -- only the one method this
    project actually calls."""

    def __init__(self, raw_client: groq.Groq) -> None:
        self._raw = raw_client

    def generate_content(self, *, model: str, contents: str, config: dict[str, Any] | None = None):
        response_format = None
        if config and config.get("response_mime_type") == "application/json":
            response_model = config["response_schema"]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _to_strict_json_schema(response_model),
                },
            }

        completion = self._raw.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": contents}],
            response_format=response_format,
        )
        choice = completion.choices[0]
        text = choice.message.content
        tokens = completion.usage.total_tokens if completion.usage is not None else None
        return _GenerateContentResult(text=text, usage_metadata=_UsageMetadata(total_token_count=tokens))


class GroqClient:
    """Drop-in replacement for `genai.Client` exposing the one attribute
    this project actually uses: `.models`. The `gemini_client` parameter
    name and `genai.Client`-shaped return type are kept everywhere else in
    the project so no other file's constructor signature or test fixture
    needed to change."""

    def __init__(self, settings: Settings) -> None:
        self._raw = groq.Groq(api_key=settings.groq_api_key, max_retries=settings.groq_retry_max_attempts)
        self.models = _GroqModelsAdapter(self._raw)


def build_gemini_client(settings: Settings) -> GroqClient:
    """Single place every LLM call site in this project constructs its
    client from -- name kept as `build_gemini_client` for minimal diff
    (see module docstring); it now builds a Groq-backed client."""
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY must be set")
    return GroqClient(settings)


def _to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Converts a Pydantic model's own model_json_schema() into the shape
    Groq's (and OpenAI's) `strict: true` structured-output mode requires:
    every object must set `additionalProperties: false` and list every
    one of its properties in `required` (Pydantic's default JSON Schema
    only lists fields without a default as required, and never sets
    additionalProperties). This adapts the *schema representation* only
    -- it does not change what the Pydantic model itself accepts, so
    `HypothesisSetResponse.model_validate_json(response.text)` and every
    other existing `model_validate_json()` call site downstream is
    unaffected."""
    schema = model.model_json_schema()
    _harden_schema_node(schema)
    return schema


def _harden_schema_node(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
        for value in node.values():
            _harden_schema_node(value)
    elif isinstance(node, list):
        for item in node:
            _harden_schema_node(item)


REQUIRED_SECTIONS = [
    "INCIDENT TYPE",
    "SEVERITY",
    "WHAT HAPPENED",
    "WHY THIS SEVERITY",
    "POTENTIAL BUSINESS IMPACT",
    "IMPORTANT SIGNALS",
    "INVESTIGATION PRIORITIES",
    "INITIAL ASSESSMENT",
]

VALID_SEVERITIES = {"low", "medium", "high", "critical"}

# Tolerate minor formatting the model might add despite instructions not
# to (markdown bold, a trailing colon) without accepting arbitrary text as
# a heading.
_HEADING_PATTERN = re.compile(
    r"^[ \t]*\**[ \t]*(" + "|".join(re.escape(h) for h in REQUIRED_SECTIONS) + r")[ \t]*:?[ \t]*\**[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


class TriageParseError(ValueError):
    """Raised when the model's text response is missing/malformed sections."""


@dataclass(frozen=True)
class TriageAssessment:
    raw_text: str
    incident_type: str
    severity: str  # normalized: low | medium | high | critical
    sections: dict[str, str]  # canonical heading -> section body text


def parse_triage_text(text: str) -> TriageAssessment:
    if not text or not text.strip():
        raise TriageParseError("Empty response from the model")

    matches = list(_HEADING_PATTERN.finditer(text))
    found = {match.group(1).strip().upper() for match in matches}
    missing = [h for h in REQUIRED_SECTIONS if h not in found]
    if missing:
        raise TriageParseError(f"Missing required section(s): {', '.join(missing)}")

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        canonical = match.group(1).strip().upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[canonical] = text[start:end].strip()

    empty = [h for h in REQUIRED_SECTIONS if not sections.get(h)]
    if empty:
        raise TriageParseError(f"Section(s) returned empty: {', '.join(empty)}")

    incident_type_line = sections["INCIDENT TYPE"].splitlines()[0].strip()
    incident_type = incident_type_line.lower().replace(" ", "_")
    if not (3 <= len(incident_type) <= 128):
        raise TriageParseError(f"Invalid INCIDENT TYPE extracted: {incident_type_line!r}")

    severity_line = sections["SEVERITY"].splitlines()[0].strip().lower()
    severity = re.sub(r"[^a-z]", "", severity_line)
    if severity not in VALID_SEVERITIES:
        raise TriageParseError(
            f"SEVERITY must be one of low/medium/high/critical, got: {sections['SEVERITY'][:80]!r}"
        )

    return TriageAssessment(
        raw_text=text.strip(),
        incident_type=incident_type,
        severity=severity,
        sections=sections,
    )


@dataclass(frozen=True)
class GeminiUsage:
    tokens_used: int | None


@dataclass(frozen=True)
class GeminiTriageResponse:
    assessment: TriageAssessment
    usage: GeminiUsage


class IntakeGenerationError(RuntimeError):
    """Raised when the model's output could not be parsed/validated even
    after one correction-prompt retry."""


class IntakeModelClient(Protocol):
    model_name: str

    def classify(self, prompt: PhaseOnePrompt) -> GeminiTriageResponse:
        ...


class GeminiIntakeClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set before running Phase 1 intake")
        self.model_name = settings.groq_model
        self.client = build_gemini_client(settings)

    def classify(self, prompt: PhaseOnePrompt) -> GeminiTriageResponse:
        try:
            return self._call(prompt.combined)
        except (TriageParseError, ValueError) as first_error:
            correction_prompt = self._correction_prompt(prompt, first_error)
            try:
                return self._call(correction_prompt)
            except (TriageParseError, ValueError) as second_error:
                raise IntakeGenerationError(
                    f"Triage output invalid after retry: {second_error}"
                ) from second_error

    def _call(self, prompt_text: str) -> GeminiTriageResponse:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
            )
        except groq.APIError as exc:
            logger.warning(
                "Triage call failed after all retry attempts (HTTP %s): %s",
                getattr(exc, "status_code", None), exc,
            )
            raise
        if not response.text:
            raise ValueError("Empty response from the model")
        assessment = parse_triage_text(response.text)
        usage_metadata = getattr(response, "usage_metadata", None)
        tokens_used = getattr(usage_metadata, "total_token_count", None) if usage_metadata else None
        return GeminiTriageResponse(assessment=assessment, usage=GeminiUsage(tokens_used=tokens_used))

    @staticmethod
    def _correction_prompt(prompt: PhaseOnePrompt, error: Exception) -> str:
        headings_block = "\n\n".join(f"{h}\n\n[generated answer]" for h in REQUIRED_SECTIONS)
        return f"""{prompt.combined}

Your previous response could not be validated. Error: {error}

Return ONLY plain human-readable text using exactly these section
headings, each on its own line, with no markdown formatting, no JSON, and
no extra headings or commentary outside these sections:

{headings_block}

SEVERITY must be exactly one of: Low, Medium, High, Critical."""


# ---------------------------------------------------------------------------
# Phase 2: Hypothesis Agent (structured JSON output)
# ---------------------------------------------------------------------------

class HypothesisItem(BaseModel):
    statement: str = Field(min_length=10, max_length=500)
    rationale: str = Field(min_length=15, max_length=2000)
    supporting_signals: str = Field(min_length=10, max_length=1000)
    confidence: str = Field(pattern="^(low|medium|high)$")
    what_would_confirm: str = Field(min_length=10, max_length=1000)
    what_would_refute: str = Field(min_length=10, max_length=1000)
    why_ranked_here: str = Field(min_length=10, max_length=1000)

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator(
        "statement", "rationale", "supporting_signals",
        "what_would_confirm", "what_would_refute", "why_ranked_here",
    )
    @classmethod
    def collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.strip().split())


class RuledOutHypothesis(BaseModel):
    statement: str = Field(min_length=10, max_length=300)
    reason_ruled_out: str = Field(min_length=10, max_length=500)

    @field_validator("statement", "reason_ruled_out")
    @classmethod
    def collapse_whitespace(cls, value: str) -> str:
        return " ".join(value.strip().split())


class HypothesisSetResponse(BaseModel):
    hypotheses: list[HypothesisItem] = Field(min_length=2, max_length=5)
    ruled_out: list[RuledOutHypothesis] = Field(default_factory=list, max_length=5)


@dataclass(frozen=True)
class GeminiHypothesisResponse:
    hypotheses: list[HypothesisItem]
    ruled_out: list[RuledOutHypothesis]
    usage: GeminiUsage


class HypothesisModelClient(Protocol):
    model_name: str

    def generate(self, prompt: HypothesisPrompt) -> GeminiHypothesisResponse:
        ...


class GeminiHypothesisClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set before running Phase 2 hypothesis generation")
        self.model_name = settings.groq_model
        self.client = build_gemini_client(settings)

    def generate(self, prompt: HypothesisPrompt) -> GeminiHypothesisResponse:
        try:
            return self._call(prompt.combined)
        except (ValidationError, ValueError) as first_error:
            correction_prompt = self._correction_prompt(prompt, first_error)
            try:
                return self._call(correction_prompt)
            except (ValidationError, ValueError) as second_error:
                raise IntakeGenerationError(
                    f"Hypothesis output invalid after retry: {second_error}"
                ) from second_error

    def _call(self, prompt_text: str) -> GeminiHypothesisResponse:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": HypothesisSetResponse,
                },
            )
        except groq.APIError as exc:
            logger.warning(
                "Hypothesis call failed after all retry attempts (HTTP %s): %s",
                getattr(exc, "status_code", None), exc,
            )
            raise
        if not response.text:
            raise ValueError("Empty response from the model")
        parsed = HypothesisSetResponse.model_validate_json(response.text)
        usage_metadata = getattr(response, "usage_metadata", None)
        tokens_used = getattr(usage_metadata, "total_token_count", None) if usage_metadata else None
        return GeminiHypothesisResponse(
            hypotheses=parsed.hypotheses,
            ruled_out=parsed.ruled_out,
            usage=GeminiUsage(tokens_used=tokens_used),
        )

    @staticmethod
    def _correction_prompt(prompt: HypothesisPrompt, error: Exception) -> str:
        return f"""{prompt.combined}

Your previous response could not be validated against the required JSON
schema. Error: {error}

Return ONLY valid JSON matching exactly this shape, with 2 to 5 items in
"hypotheses" and 0 to 5 items in "ruled_out", no markdown, and no extra
keys:
{{
    "hypotheses": [
        {{
            "statement": "...",
            "rationale": "...",
            "supporting_signals": "...",
            "confidence": "low|medium|high",
            "what_would_confirm": "...",
            "what_would_refute": "...",
            "why_ranked_here": "..."
        }}
    ],
    "ruled_out": [
        {{
            "statement": "...",
            "reason_ruled_out": "..."
        }}
    ]
}}"""
