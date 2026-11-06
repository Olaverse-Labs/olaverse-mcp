# Olaverse MCP server

[Olaverse Lab](https://www.olaverselab.com)'s API as tools for AI agents: Claude, Cursor and any other MCP client.
The models are small, task-specific and strongest on English and African-language text.

| Tool | What it does |
|---|---|
| `extract_document` | A document (text, a JPEG/PNG photo or scan, or a PDF) to JSON shaped like your template, with a calibrated confidence for every field. Long documents are split, extracted in parallel and merged. |
| `get_extraction_job` | Collects a long-document result that's still running. |
| `decide` | Typed questions about a message, ticket or record (yes/no, choice, score), with calibrated probabilities. 19 languages. |
| `detect_language` | The language of a text, among 608 (built African-first, good on very short text). |
| `restore_diacritics` | Puts back tone marks, accents and Arabic vowel marks in 11 languages, including Yorùbá, Igbo, Hausa and Arabic. |
| `generate_questions` | Search-style questions a passage answers, in the passage's own language (25 languages). |

## Set up

You need an Olaverse API key: create one at [console.olaverselab.com](https://console.olaverselab.com) (new
accounts get $1 of free credit). Calls are billed to your key at the [usual prices](https://docs.olaverselab.com).

The server runs with [uv](https://docs.astral.sh/uv/) (`uvx` downloads and runs it).

**Claude Code**

```bash
claude mcp add olaverse --env OLAVERSE_API_KEY=your-key -- uvx olaverse-mcp
```

**Claude Desktop, Cursor and others** (the `mcpServers` block of the client's MCP config file)

```json
{
  "mcpServers": {
    "olaverse": {
      "command": "uvx",
      "args": ["olaverse-mcp"],
      "env": { "OLAVERSE_API_KEY": "your-key" }
    }
  }
}
```

**Hosted (no install).** Olaverse also runs this server at `https://mcp.olaverselab.com/mcp` (streamable HTTP).
Send your key with each request as `Authorization: Bearer your-key`. The hosted server takes documents as `text`,
or as an `image` or `pdf` in base64; it doesn't read files or fetch URLs.

## Use it

Ask in plain words, and the agent picks the tool:

- "Extract the merchant, date, total and line items from ~/Downloads/receipt.jpg."
- "Is this customer message asking for a refund, and how urgent is it? «Ẹ jọ̀ọ́, wọ́n ti yọ owó lẹ́ẹ̀mejì…»"
- "Which language is 'Mo fẹ́ lọ sí ọjà'?"
- "Add the tone marks back: se eranko naa si gbo o?"

**Confidence.** `extract_document` returns a confidence for every field. It's calibrated: fields at 0.95 and
above are right about 95% of the time on the documents it was built for. Check a threshold on a sample of your
own documents before you automate on it.

## Settings

| Variable | Default | |
|---|---|---|
| `OLAVERSE_API_KEY` | required | Your API key. |
| `OLAVERSE_API_URL` | `https://api.olaverselab.com` | |
| `OLAVERSE_JOB_WAIT` | `900` | Seconds a long-document extraction waits before handing back a job id for `get_extraction_job`. |

Scanned PDFs (no text layer) aren't read directly: send their pages as JPEG or PNG images.

## Licence

Apache-2.0.

<!-- MCP registry ownership: -->
mcp-name: io.github.Olaverse-Labs/olaverse-mcp
