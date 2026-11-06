"""Olaverse Lab's API as MCP tools: document extraction, decisions, language ID, diacritics, question generation.

Runs over stdio (Claude Desktop, Claude Code, Cursor and other MCP clients start it for you). Needs an Olaverse API
key in OLAVERSE_API_KEY (console.olaverselab.com); calls are billed to that key like any other API call.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

API = os.environ.get("OLAVERSE_API_URL", "https://api.olaverselab.com").rstrip("/")
PLAIN_LIMIT = 60_000          # characters /v1/extract takes in one call; longer documents go to a job
JOB_WAIT = float(os.environ.get("OLAVERSE_JOB_WAIT", 900))   # seconds a tool call waits for a long-document job
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

server = MCPServer(
    name="olaverse",
    title="Olaverse Lab",
    website_url="https://www.olaverselab.com",
    instructions=(
        "Olaverse Lab's models, strongest on English and African-language text. extract_document turns a document "
        "(text, a JPEG/PNG photo or scan, or a PDF file) into JSON from a template and gives a calibrated confidence "
        "for every field: accept fields at 0.95 or above, check the rest. decide answers typed questions about a "
        "message or record with calibrated probabilities. detect_language covers 608 languages, restore_diacritics "
        "puts back tone marks and accents in 11 languages, generate_questions writes search-style questions a "
        "passage answers, in the passage's own language. Every call is billed to the user's API key."),
)


class OlaverseError(ToolError):
    """A problem the agent should see and can act on (bad input, no key, an API error): shown as the tool's error."""


def _key() -> str:
    key = os.environ.get("OLAVERSE_API_KEY", "").strip()
    if not key:
        raise OlaverseError("Set OLAVERSE_API_KEY to an Olaverse API key (create one at https://console.olaverselab.com).")
    return key


def _request(method: str, path: str, body: Optional[dict] = None, timeout: float = 240) -> dict:
    try:
        r = httpx.request(method, API + path, json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {_key()}", "User-Agent": "olaverse-mcp"})
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


def _post(path: str, body: dict, timeout: float = 240) -> dict:
    return _request("POST", path, body, timeout)


def _read_document(file_path: str) -> dict:
    """{"text": ...} for a PDF (its text layer, pages separated by form feeds) or a text file; {"image": data URL}
    for a JPEG or PNG; {"pages": [...]} for a PDF with no text layer, as page images."""
    p = Path(file_path).expanduser()
    if not p.is_file():
        raise OlaverseError(f"No such file: {p}")
    suffix = p.suffix.lower()
    if suffix in IMAGE_TYPES:
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        return {"image": f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()}
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        pages = [(page.extract_text() or "") for page in reader.pages]
        if sum(len(t.strip()) for t in pages) >= 100 * max(1, len(pages)):
            return {"text": "\f".join(pages)}
        raise OlaverseError("This PDF has no text layer (a scan). Send its pages as JPEG or PNG images instead, one per call.")
    return {"text": p.read_text(errors="replace")}


def _run_job(body: dict) -> dict:
    job = _post("/v1/extract/jobs", body)
    deadline = time.time() + JOB_WAIT
    while time.time() < deadline:
        time.sleep(5)
        r = _request("GET", job["poll"], timeout=60)
        if r.get("status") != "running":
            if r.get("status") == "failed":
                raise OlaverseError(f"The job failed: {r.get('error')}")
            return {**r, "job": True}
    return {"id": job["id"], "status": "running", "parts": job.get("parts"), "estimate": job.get("estimate"),
            "note": f"Still running after {JOB_WAIT:.0f} s: call get_extraction_job with this id to collect the result."}


@server.tool()
def extract_document(template: dict[str, Any], text: Optional[str] = None, file_path: Optional[str] = None,
                     instructions: Optional[str] = None) -> dict:
    """Turn a document into JSON shaped like `template`, with a calibrated confidence (0-1) for every field.

    template: the shape of the answer; each leaf names a type: "verbatim-string" (copied exactly), "string",
      "number", "integer", "date" (YYYY-MM-DD), "date-time", "boolean", ["a", "b"] (pick one), [["a", "b"]] (pick
      any), ["string"] (a list), or [{...}] (a list of objects, e.g. line items).
      Example: {"merchant": "verbatim-string", "total": "number", "date": "date",
                "items": [{"name": "verbatim-string", "price": "number"}]}
    text: the document as text. Or give file_path instead: a JPEG/PNG photo or scan, a PDF with a text layer, or a
      text file on this machine.
    instructions: optional guidance, e.g. "dates are day/month/year" or "amounts in kobo".

    Long documents (over 60,000 characters) are split into parts, extracted in parallel and merged; that can take
    minutes. Missing values come back as null. Treat fields with confidence below about 0.95 as needing a check.
    """
    if bool(text) == bool(file_path):
        raise OlaverseError("Give either `text` or `file_path`.")
    doc = {"text": text} if text else _read_document(file_path)
    body = {"template": template, **doc, **({"instructions": instructions} if instructions else {})}
    if len(doc.get("text") or "") > PLAIN_LIMIT:
        return _run_job(body)
    r = _post("/v1/extract", body)
    return {k: r.get(k) for k in ("data", "confidence", "usage", "calibration", "error") if r.get(k) is not None}


@server.tool()
def get_extraction_job(job_id: str) -> dict:
    """The result of a long-document extraction job that extract_document left running (by its id, exj_...)."""
    return _request("GET", f"/v1/extract/jobs/{job_id}", timeout=60)


@server.tool()
def decide(state: Any, questions: dict[str, Any]) -> dict:
    """Answer typed questions about a message, ticket, record or form, with a calibrated probability for each answer.

    state: the thing to decide about: a string (e.g. a customer message) or a JSON object (a record).
    questions: {question_id: {...}}, each one of:
      {"type": "noul", "instructions": "The customer is asking for a refund."}         -> probability it's true
      {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "delivery": "orders"}}
      {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today", "immediately"]}
    Works in 19 languages, including English, Pidgin, Hausa, Yorùbá, Igbo, Swahili and Arabic.
    """
    return _post("/v1/decide", {"state": state, "questions": questions})


@server.tool()
def detect_language(text: str, mode: str = "traffic", top_k: int = 3) -> dict:
    """Identify the language of a text among 608 languages (built African-first; good on very short text).

    mode: "traffic" (user messages and chat; favours languages people actually use) or "coverage" (all languages
    equal, for building corpora). Returns the ISO 639-3 code, script, name, confidence and runners-up.
    """
    return _post("/v1/language-id", {"text": text, "mode": mode, "top_k": top_k})


@server.tool()
def restore_diacritics(text: str, language: str) -> dict:
    """Put back diacritics, tone marks or Arabic vowel marks in text typed without them.

    language: yor (Yorùbá), ibo (Igbo), hau (Hausa), vie, pol, tur, por, spa, fra, ita or ara (Arabic).
    Example: "se eranko naa si gbo o?" (yor) -> "ṣé ẹranko náà sì gbọ́ ọ?"
    """
    return _post("/v1/diacritize", {"text": text, "language": language})


@server.tool()
def generate_questions(passage: str, n: int = 3, language: Optional[str] = None) -> dict:
    """Write n natural, search-style questions that a passage answers, in the passage's own language (25 languages).

    Useful for FAQs, study cards and search or RAG training data. passage: up to 8,000 characters.
    """
    return _post("/v1/questions/generate", {"passage": passage, "n": n, **({"language": language} if language else {})})


def main():
    server.run()


if __name__ == "__main__":
    main()
