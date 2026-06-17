from __future__ import annotations

"""
The ONLY place the frontend makes HTTP calls to the backend.

Uses httpx.AsyncClient. The Streamlit UI imports this class and never builds raw
requests itself — keeping transport concerns out of the view layer.
"""

import os

import httpx

_DEFAULT_BASE_URL = os.getenv("CHATBOT_BACKEND_URL", "http://localhost:8000")
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class ChatAPIClient:
    """Thin async wrapper around the chatbot backend's HTTP API."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def send_collect_message(
        self,
        history: list[dict],
        message: str,
    ) -> dict:
        """POST /api/chat/collect → {reply, user_info|null, phase}."""
        payload = {"user_message": message, "conversation_history": history}
        return await self._post("/api/chat/collect", payload)

    async def send_qa_message(
        self,
        history: list[dict],
        user_info: dict,
        message: str,
    ) -> dict:
        """POST /api/chat/qa → {reply}."""
        payload = {
            "user_message": message,
            "conversation_history": history,
            "user_info": user_info,
        }
        return await self._post("/api/chat/qa", payload)

    async def _post(self, path: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(f"{self.base_url}{path}", json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                detail = _extract_detail(exc.response)
                raise ChatAPIError(detail) from exc
            except httpx.HTTPError as exc:
                raise ChatAPIError(
                    f"Could not reach the chatbot backend at {self.base_url}. "
                    f"Is it running? ({exc})"
                ) from exc


class ChatAPIError(Exception):
    """Raised when the backend returns an error or is unreachable."""


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:
        pass
    return f"Backend returned HTTP {response.status_code}."
