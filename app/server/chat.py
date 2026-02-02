import asyncio
import ssl
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
    import httpcore
except ImportError:
    httpx = None
    httpcore = None

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from loguru import logger

from ..models import (
    ChatCompletionRequest,
    ConversationInStore,
    Message,
    ModelData,
    ModelListResponse,
)
from ..services import (
    GeminiClientPool,
    GeminiClientWrapper,
    LMDBConversationStore,
)
from ..utils import g_config
from ..utils.helper import estimate_tokens
from .middleware import get_temp_dir, verify_api_key

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"


router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    now = int(datetime.now(tz=timezone.utc).timestamp())

    models = []
    for model in Model:
        m_name = model.model_name
        if not m_name or m_name == "unspecified":
            continue

        models.append(
            ModelData(
                id=m_name,
                created=now,
                owned_by="gemini-web",
            )
        )

    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    pool = GeminiClientPool()
    db = LMDBConversationStore()
    model = Model.from_name(request.model)

    if len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required in the conversation.",
        )

    # Check if conversation is reusable
    session, client, remaining_messages = _find_reusable_session(db, pool, model, request.messages)

    if session:
        # Prepare the model input depending on how many turns are missing.
        if len(remaining_messages) == 1:
            model_input, files = await GeminiClientWrapper.process_message(
                remaining_messages[0], tmp_dir, tagged=False
            )
        else:
            model_input, files = await GeminiClientWrapper.process_conversation(
                remaining_messages, tmp_dir
            )
        logger.debug(
            f"Reused session {session.metadata} - sending {len(remaining_messages)} new messages."
        )
    else:
        # Start a new session and concat messages into a single string
        try:
            client = pool.acquire()  # Logging happens in acquire()
            session = client.start_chat(model=model)
            logger.info(f"💬 Chat completion request using client: {client.id}")
            model_input, files = await GeminiClientWrapper.process_conversation(
                request.messages, tmp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
        logger.debug("New session started.")

    # Generate response
    try:
        assert session and client, "Session and client not available"
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        response = await _send_with_split(session, model_input, files=files)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # Format the response from API
    model_output = GeminiClientWrapper.extract_output(response, include_thoughts=True)
    stored_output = GeminiClientWrapper.extract_output(response, include_thoughts=False)

    # After formatting, persist the conversation to LMDB
    try:
        last_message = Message(role="assistant", content=stored_output)
        cleaned_history = db.sanitize_assistant_messages(request.messages)
        conv = ConversationInStore(
            model=model.model_name,
            client_id=client.id,
            metadata=session.metadata,
            messages=[*cleaned_history, last_message],
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key}")
    except Exception as e:
        # We can still return the response even if saving fails
        logger.warning(f"Failed to save conversation to LMDB: {e}")

    # Return with streaming or standard response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(datetime.now(tz=timezone.utc).timestamp())
    if request.stream:
        return _create_streaming_response(
            model_output,
            completion_id,
            timestamp,
            request.model,
            request.messages,
        )
    else:
        return _create_standard_response(
            model_output, completion_id, timestamp, request.model, request.messages
        )


def _text_from_message(message: Message) -> str:
    """Return text content from a message for token estimation."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        item.text or "" for item in message.content if getattr(item, "type", "") == "text"
    )


def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[Message],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[Message]]:
    """Find an existing chat session that matches the *longest* prefix of
    ``messages`` **whose last element is an assistant/system reply**.

    Rationale
    ---------
    When a reply was generated by *another* server instance, the local LMDB may
    only contain an older part of the conversation.  However, as long as we can
    line-up **any** earlier assistant/system response, we can restore the
    corresponding Gemini session and replay the *remaining* turns locally
    (including that missing assistant reply and the subsequent user prompts).

    The algorithm therefore walks backwards through the history **one message at
    a time**, each time requiring the current tail to be assistant/system before
    querying LMDB.  As soon as a match is found we recreate the session and
    return the untouched suffix as ``remaining_messages``.
    """

    if len(messages) < 2:
        return None, None, messages

    # Start with the full history and iteratively trim from the end.
    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]

        # Only try to match if the last stored message would be assistant/system.
        if search_history[-1].role in {"assistant", "system"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = pool.acquire(conv.client_id)
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    return session, client, remain
            except Exception as e:
                logger.warning(f"Error checking LMDB for reusable session: {e}")
                break

        # Trim one message and try again.
        search_end -= 1

    return None, None, messages


async def _send_with_split(session: ChatSession, text: str, files: list[Path | str] | None = None):
    """Send text to Gemini, automatically splitting into multiple batches if it is
    longer than ``MAX_CHARS_PER_REQUEST``.

    Every intermediate batch (that is **not** the last one) is suffixed with a hint
    telling Gemini that more content will come, and it should simply reply with
    "ok". The final batch carries any file uploads and the real user prompt so
    that Gemini can produce the actual answer.
    """
    
    async def _send_with_retry(message_text: str, message_files: list[Path | str] | None = None, max_retries: int = 3):
        """Helper function to send a message with SSL error retry logic."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await session.send_message(message_text, files=message_files)
            except (ssl.SSLError, OSError) as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['ssl', 'decryption', 'bad record mac', 'connection', 'network']):
                    logger.warning(f"SSL/Network error sending message (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        await asyncio.sleep(wait_time)
                    last_exception = e
                else:
                    raise
            except Exception as e:
                # Check for httpx/httpcore exceptions that might wrap SSL errors
                if httpx:
                    httpx_exceptions = (
                        getattr(httpx, 'TimeoutException', None),
                        getattr(httpx, 'ReadTimeout', None),
                        getattr(httpx, 'ConnectError', None),
                        getattr(httpx, 'RemoteProtocolError', None),
                    )
                    httpx_exceptions = tuple(exc for exc in httpx_exceptions if exc is not None)
                    if httpx_exceptions and isinstance(e, httpx_exceptions):
                        logger.warning(f"HTTP error sending message (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                        last_exception = e
                        continue
                
                if httpcore:
                    httpcore_exceptions = (
                        getattr(httpcore, 'ReadTimeout', None),
                        getattr(httpcore, 'ConnectError', None),
                        getattr(httpcore, 'RemoteProtocolError', None),
                    )
                    httpcore_exceptions = tuple(exc for exc in httpcore_exceptions if exc is not None)
                    if httpcore_exceptions and isinstance(e, httpcore_exceptions):
                        logger.warning(f"HTTPCore error sending message (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                        last_exception = e
                        continue
                
                # Check if it's an SSL-related error in the exception message
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['ssl', 'decryption', 'bad record mac', 'tls']):
                    logger.warning(f"SSL-related error detected (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    last_exception = e
                else:
                    raise
        
        if last_exception:
            raise last_exception
    
    if len(text) <= MAX_CHARS_PER_REQUEST:
        # No need to split - a single request is fine.
        return await _send_with_retry(text, files)
    
    hint_len = len(CONTINUATION_HINT)
    chunk_size = MAX_CHARS_PER_REQUEST - hint_len

    chunks: list[str] = []
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = text[pos:end]
        pos = end

        # If this is NOT the last chunk, add the continuation hint.
        if end < total:
            chunk += CONTINUATION_HINT
        chunks.append(chunk)

    # Fire off all but the last chunk, discarding the interim "ok" replies.
    for chk in chunks[:-1]:
        await _send_with_retry(chk)

    # The last chunk carries the files (if any) and we return its response.
    return await _send_with_retry(chunks[-1], files)


def _create_streaming_response(
    model_output: str,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> StreamingResponse:
    """Create streaming response with `usage` calculation included in the final chunk."""

    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    async def generate_stream():
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Stream output text in chunks for efficiency
        chunk_size = 32
        for i in range(0, len(model_output), chunk_size):
            chunk = model_output[i : i + chunk_size]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Send end event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    model_output: str,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> dict:
    """Create standard response"""
    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": model_output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.debug(f"Response created with {total_tokens} total tokens")
    return result
