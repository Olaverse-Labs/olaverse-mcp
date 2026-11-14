"""Olaverse Lab's API as MCP tools: document extraction, decisions, language ID, diacritics, question generation.

Two ways to run it:
- Local (stdio): `uvx olaverse-mcp`, started by Claude Desktop, Claude Code, Cursor and other clients, with the API
  key in OLAVERSE_API_KEY. It can read documents from local files.
- Hosted (streamable HTTP): `http_app()`, served by Olaverse at https://mcp.olaverselab.com/mcp. Each request brings
  the caller's own API key (Authorization: Bearer <key>, or X-Olaverse-Key), which every tool call uses, so calls are
  billed to that key. Documents come as text or base64; the hosted server never reads files or fetches URLs.
"""
from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import os
import asyncio
import time
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

API = os.environ.get("OLAVERSE_API_URL", "https://api.olaverselab.com").rstrip("/")
PLAIN_LIMIT = 60_000          # characters /v1/extract takes in one call; longer documents go to a job
JOB_WAIT = float(os.environ.get("OLAVERSE_JOB_WAIT", 900))   # seconds a tool call waits for a long-document job
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
HOSTED = False                # set by http_app(): no local files, no environment key

server = MCPServer(
    name="olaverse",
    title="Olaverse Lab",
    website_url="https://www.olaverselab.com",
    instructions=(
        "Olaverse Lab's specialist models for many languages, including the ones others skip. extract_document turns a document "
        "(text, a JPEG/PNG image, or a PDF) into JSON from a template and gives a calibrated confidence for every "
        "field: accept fields at 0.95 or above, check the rest. decide answers typed questions about a message or "
        "record with calibrated probabilities. detect_language covers 608 languages, restore_diacritics puts back "
        "tone marks and accents in 11 languages, generate_questions writes the questions a passage answers, in 324 "
        "languages, in its own language or another. Every call is billed to the user's API key."),
)


class OlaverseError(ToolError):
    """A problem the agent should see and can act on (bad input, no key, an API error): shown as the tool's error."""


def _key(ctx: Optional[Context]) -> str:
    """The caller's API key: from the request's headers when hosted, else from OLAVERSE_API_KEY."""
    headers = None
    try:
        headers = ctx.headers if ctx is not None else None
    except ValueError:   # no request context (called directly)
        headers = None
    if headers:
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else (headers.get("x-olaverse-key") or "").strip()
        if key:
            return key
    if not HOSTED:
        key = os.environ.get("OLAVERSE_API_KEY", "").strip()
        if key:
            return key
    raise OlaverseError("No Olaverse API key: send it as 'Authorization: Bearer <key>' (hosted), or set "
                        "OLAVERSE_API_KEY (local). Create one at https://console.olaverselab.com.")


def _check_key_shape(key: str):
    """Hosted mode: refuse a key that can't be an Olaverse key (a signed token, header.payload.signature, unexpired)
    before calling the API. The hosted server shares one outbound address, and the API bans an address after a run
    of failed logins, so mistyped keys must never reach it."""
    import json as _json
    parts = key.split(".")
    try:
        if len(parts) != 3:
            raise ValueError
        payload = _json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, binascii.Error):
        raise OlaverseError("That isn't an Olaverse API key: copy the key from https://console.olaverselab.com (it starts with 'eyJ').")
    if isinstance(payload, dict) and isinstance(payload.get("exp"), (int, float)) and payload["exp"] < time.time():
        raise OlaverseError("This Olaverse API key has expired: create a new one at https://console.olaverselab.com.")


async def _request(ctx: Optional[Context], method: str, path: str, body: Optional[dict] = None, timeout: float = 240) -> dict:
    key = _key(ctx)
    if HOSTED:
        _check_key_shape(key)
    try:   # async: the hosted server shares an event loop with other requests, so it must never block it
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, API + path, json=body,
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": "olaverse-mcp"})
    except httpx.HTTPError as e:
        raise OlaverseError(f"Could not reach the Olaverse API ({type(e).__name__}); try again.") from e
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400:
        err = data.get("error") if isinstance(data, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else None) or (data.get("detail") if isinstance(data, dict) else None) \
            or (data.get("message") if isinstance(data, dict) else None) or r.text[:300]
        raise OlaverseError(f"The Olaverse API returned {r.status_code}: {msg}")
    return data


def _pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader
    pages = [(page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages]
    if sum(len(t.strip()) for t in pages) < 100 * max(1, len(pages)):
        raise OlaverseError("This PDF has no text layer (a scan). Send its pages as JPEG or PNG images instead, one per call.")
    return "\f".join(pages)   # page breaks: the API keeps whole pages together when it splits long documents


def _b64(value: str, what: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise OlaverseError(f"'{what}' must be base64 (or a data URL).") from e


def _read_file(file_path: str) -> dict:
    if HOSTED:
        raise OlaverseError("The hosted server can't read files: send the document as 'text', 'image' or 'pdf' (base64).")
    p = Path(file_path).expanduser()
    if not p.is_file():
        raise OlaverseError(f"No such file: {p}")
    suffix = p.suffix.lower()
    if suffix in IMAGE_TYPES:
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        return {"image": f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()}
    if suffix == ".pdf":
        return {"text": _pdf_text(p.read_bytes())}
    return {"text": p.read_text(errors="replace")}


async def _run_job(ctx: Optional[Context], body: dict) -> dict:
    job = await _request(ctx, "POST", "/v1/extract/jobs", body)
    deadline = time.time() + JOB_WAIT
    while time.time() < deadline:
        await asyncio.sleep(5)
        r = await _request(ctx, "GET", job["poll"], timeout=60)
        if r.get("status") != "running":
            if r.get("status") == "failed":
                raise OlaverseError(f"The job failed: {r.get('error')}")
            return {**r, "job": True}
    return {"id": job["id"], "status": "running", "parts": job.get("parts"), "estimate": job.get("estimate"),
            "note": f"Still running after {JOB_WAIT:.0f} s: call get_extraction_job with this id to collect the result."}


@server.tool()
async def extract_document(template: dict[str, Any], text: Optional[str] = None, image: Optional[str] = None,
                     pdf: Optional[str] = None, file_path: Optional[str] = None, instructions: Optional[str] = None,
                     ctx: Context = None) -> dict:
    """Turn a document into JSON shaped like `template`, with a calibrated confidence (0-1) for every field.

    template: the shape of the answer; each leaf names a type: "verbatim-string" (copied exactly), "string",
      "number", "integer", "date" (YYYY-MM-DD), "date-time", "boolean", ["a", "b"] (pick one), [["a", "b"]] (pick
      any), ["string"] (a list), or [{...}] (a list of objects, e.g. line items).
      Example: {"merchant": "verbatim-string", "total": "number", "date": "date",
                "items": [{"name": "verbatim-string", "price": "number"}]}
    Give the document as exactly one of:
      text: the document as text.
      image: a JPEG or PNG, base64 or a data URL (a photo, scan or screenshot of one page).
      pdf: a PDF with a text layer, base64.
      file_path: a JPEG/PNG, PDF or text file on this machine (local server only).
    instructions: optional guidance, e.g. "dates are day/month/year" or "amounts in kobo".

    Long documents (over 60,000 characters) are split into parts, extracted in parallel and merged; that can take
    minutes. Missing values come back as null. Treat fields with confidence below about 0.95 as needing a check.
    """
    given = [k for k, v in (("text", text), ("image", image), ("pdf", pdf), ("file_path", file_path)) if v]
    if len(given) != 1:
        raise OlaverseError("Give the document as exactly one of 'text', 'image', 'pdf' or 'file_path'.")
    if text:
        doc = {"text": text}
    elif image:
        doc = {"image": image if image.startswith("data:") else "data:image/jpeg;base64," + base64.b64encode(_b64(image, "image")).decode()}
    elif pdf:
        doc = {"text": _pdf_text(_b64(pdf, "pdf"))}
    else:
        doc = _read_file(file_path)
    body = {"template": template, **doc, **({"instructions": instructions} if instructions else {})}
    if len(doc.get("text") or "") > PLAIN_LIMIT:
        return await _run_job(ctx, body)
    r = await _request(ctx, "POST", "/v1/extract", body)
    return {k: r.get(k) for k in ("data", "confidence", "usage", "calibration", "error") if r.get(k) is not None}


@server.tool()
async def get_extraction_job(job_id: str, ctx: Context = None) -> dict:
    """The result of a long-document extraction job that extract_document left running (by its id, exj_...)."""
    return await _request(ctx, "GET", f"/v1/extract/jobs/{job_id}", timeout=60)


@server.tool()
async def decide(state: Any, questions: dict[str, Any], ctx: Context = None) -> dict:
    """Answer typed questions about a message, ticket, record or form, with a calibrated probability for each answer.

    state: the thing to decide about: a string (e.g. a customer message) or a JSON object (a record).
    questions: {question_id: {...}}, each one of:
      {"type": "noul", "instructions": "The customer is asking for a refund."}         -> probability it's true
      {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "delivery": "orders"}}
      {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today", "immediately"]}
    Works in 19 languages, including English, Pidgin, Hausa, Yorùbá, Igbo, Swahili and Arabic.
    """
    return await _request(ctx, "POST", "/v1/decide", {"state": state, "questions": questions})


@server.tool()
async def detect_language(text: str, mode: str = "traffic", top_k: int = 3, ctx: Context = None) -> dict:
    """Identify the language of a text among 608 languages (built African-first; good on very short text).

    mode: "traffic" (user messages and chat; favours languages people actually use) or "coverage" (all languages
    equal, for building corpora). Returns the ISO 639-3 code, script, name, confidence and runners-up.
    """
    return await _request(ctx, "POST", "/v1/language-id", {"text": text, "mode": mode, "top_k": top_k})


@server.tool()
async def restore_diacritics(text: str, language: str, ctx: Context = None) -> dict:
    """Put back diacritics, tone marks or Arabic vowel marks in text typed without them.

    language: yor (Yorùbá), ibo (Igbo), hau (Hausa), vie, pol, tur, por, spa, fra, ita or ara (Arabic).
    Example: "se eranko naa si gbo o?" (yor) -> "ṣé ẹranko náà sì gbọ́ ọ?"
    """
    return await _request(ctx, "POST", "/v1/diacritize", {"text": text, "language": language})


@server.tool()
async def generate_questions(passage: str, n: Union[int, str] = 3, language: Optional[str] = None,
                             question_language: Optional[str] = None, ctx: Context = None) -> dict:
    """Write the questions a passage answers (MIST-QG-1.1, passages in 324 languages).

    Useful for FAQs, study cards and search or RAG training data. passage: up to 8,000 characters, best under 2,000.
    n: 1 to 5, or "auto" to write as many as the passage supports. language: the passage's language (detected when
    left out). question_language: write the questions in another language: English or French for any passage, or one
    of 30 widely spoken languages (Spanish, Hindi, Chinese, Swahili, Yoruba, Amharic...) for an English or French
    passage or one in those 30.
    """
    body = {"passage": passage, "n": n}
    if language:
        body["language"] = language
    if question_language:
        body["question_language"] = question_language
    return await _request(ctx, "POST", "/v1/questions/generate", body)


def http_app(allowed_hosts: Optional[list] = None, path: str = "/mcp"):
    """The hosted server as an ASGI app (streamable HTTP, stateless, so any replica can answer any request). The
    caller's key comes with each request; files and environment keys are off. Run its session manager for the
    app's lifetime: `async with server.session_manager.run(): ...` (Starlette's lifespan does it when served alone)."""
    from mcp.server.transport_security import TransportSecuritySettings

    global HOSTED
    HOSTED = True
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(allowed_hosts),
                                         allowed_hosts=allowed_hosts or [], allowed_origins=[])
    return server.streamable_http_app(streamable_http_path=path, stateless_http=True, json_response=True,
                                      max_request_body_size=40_000_000, transport_security=security)


def main():
    server.run()


if __name__ == "__main__":
    main()
