from __future__ import annotations

"""
FastAPI app for the stateless HMO medical-services chatbot.

Endpoints:
  POST /api/chat/collect — advance the info-collection conversation
  POST /api/chat/qa      — answer a question from the user's HMO knowledge
  GET  /health           — liveness + index status

The service is stateless: every request carries the full conversation history
(and, for Q&A, the confirmed user_info). The ADA-002 knowledge index is the only
process-level state and is built once at startup.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.logger import get_logger
from part2.backend import chat_service
from part2.backend import knowledge_base as kb

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class UserInfo(BaseModel):
    firstName: str = ""
    lastName: str = ""
    idNumber: str = ""
    gender: str = ""
    age: Union[int, str] = ""
    hmo: str = ""
    hmoCardNumber: str = ""
    insuranceTier: str = ""


class CollectRequest(BaseModel):
    user_message: str
    conversation_history: List[Message] = Field(default_factory=list)


class CollectResponse(BaseModel):
    reply: str
    user_info: Optional[UserInfo] = None
    phase: str


class QARequest(BaseModel):
    user_message: str
    conversation_history: List[Message] = Field(default_factory=list)
    user_info: UserInfo


class QAResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Lifespan — build the knowledge index once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building knowledge index at startup…")
    try:
        await kb.build_index()
    except Exception:
        logger.exception("Failed to build knowledge index")
        raise
    yield
    logger.info("Shutting down chatbot service")


app = FastAPI(
    title="HMO Medical-Services Chatbot",
    description="Stateless chatbot for Maccabi / Meuhedet / Clalit medical services.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "knowledge_topics": len(kb.topic_index),
        "index_ready": bool(kb.topic_index),
    }


@app.post("/api/chat/collect", response_model=CollectResponse)
async def collect(body: CollectRequest) -> CollectResponse:
    if not body.user_message.strip():
        raise HTTPException(status_code=422, detail="user_message must not be empty.")
    try:
        result = await chat_service.run_collection(
            history=[m.model_dump() for m in body.conversation_history],
            user_message=body.user_message,
        )
    except Exception as exc:
        logger.exception("Collection endpoint failed")
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc
    return CollectResponse(**result)


@app.post("/api/chat/qa", response_model=QAResponse)
async def qa(body: QARequest) -> QAResponse:
    if not body.user_message.strip():
        raise HTTPException(status_code=422, detail="user_message must not be empty.")
    if not str(body.user_info.hmo).strip():
        raise HTTPException(
            status_code=422,
            detail="user_info.hmo is required for the Q&A phase.",
        )
    try:
        result = await chat_service.run_qa(
            history=[m.model_dump() for m in body.conversation_history],
            user_info=body.user_info.model_dump(),
            user_message=body.user_message,
        )
    except Exception as exc:
        logger.exception("Q&A endpoint failed")
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc
    return QAResponse(**result)
