import json
from collections.abc import Iterator

import anthropic

from app.config import settings
from app.retrieval import retrieve

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY "
    "the information in the provided documents. If the documents don't contain "
    "the answer, say so plainly instead of guessing. Be concise."
)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _build_content(question: str, chunks: list) -> list[dict]:
    # Each retrieved chunk becomes its own cited document block - Claude's
    # citations API grounds every claim in the response back to a specific
    # chunk, instead of relying on the model to self-report sources in text.
    content = [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": chunk.page_content,
            },
            "title": chunk.metadata.get("source", "document"),
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]
    content.append({"type": "text", "text": question})
    return content


def _sse(event_type: str, payload) -> str:
    return f"data: {json.dumps({'type': event_type, 'data': payload})}\n\n"


def answer_question_stream(question: str, top_k: int | None = None) -> Iterator[str]:
    """Server-Sent-Events generator: 'token' deltas, then one 'done' with citations."""
    chunks = retrieve(question, top_k)
    if not chunks:
        yield _sse("token", "No documents have been ingested yet - upload some first.")
        yield _sse("done", {"citations": []})
        return

    content = _build_content(question, chunks)
    client = get_client()

    try:
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield _sse("token", event.delta.text)
            final = stream.get_final_message()
    except anthropic.APIStatusError as e:
        yield _sse("error", f"Claude API error: {e.message}")
        return
    except anthropic.APIConnectionError:
        yield _sse("error", "Could not reach the Claude API. Check your connection.")
        return

    if final.stop_reason == "refusal":
        yield _sse("error", "Claude declined to answer this question.")
        return

    citations = []
    for block in final.content:
        if block.type == "text" and block.citations:
            for c in block.citations:
                citations.append(
                    {
                        "source": getattr(c, "document_title", None) or "document",
                        "cited_text": c.cited_text,
                    }
                )

    yield _sse("done", {"citations": citations})
