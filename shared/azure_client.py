"""
Single source of truth for all Azure SDK client singletons.

All other modules import clients from here — never instantiate AzureOpenAI,
AsyncAzureOpenAI, or DocumentAnalysisClient elsewhere.

Raises ValueError at import time if any required environment variable is missing,
so misconfiguration fails fast on startup rather than mid-request.
"""

import os

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            "Copy .env.example → .env and fill in the values."
        )
    return value


# --- Configuration (read once at import time) ---

_OPENAI_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT")
_OPENAI_KEY = _require("AZURE_OPENAI_KEY")
_OPENAI_API_VERSION = _require("AZURE_OPENAI_API_VERSION")

_DOC_INTEL_ENDPOINT = _require("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
_DOC_INTEL_KEY = _require("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Deployment names — imported by callers alongside the client singletons
GPT4O_DEPLOYMENT: str = _require("AZURE_OPENAI_GPT4O_DEPLOYMENT")
GPT4O_MINI_DEPLOYMENT: str = _require("AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT")
ADA_DEPLOYMENT: str = _require("AZURE_OPENAI_ADA_DEPLOYMENT")


# --- Client singletons ---

# Synchronous — used by Part 1 (Streamlit, single-user, no async needed)
openai_client: AzureOpenAI = AzureOpenAI(
    azure_endpoint=_OPENAI_ENDPOINT,
    api_key=_OPENAI_KEY,
    api_version=_OPENAI_API_VERSION,
)

# Asynchronous — used by Part 2 (FastAPI, concurrent users, all calls are awaited)
async_openai_client: AsyncAzureOpenAI = AsyncAzureOpenAI(
    azure_endpoint=_OPENAI_ENDPOINT,
    api_key=_OPENAI_KEY,
    api_version=_OPENAI_API_VERSION,
)

# Document Intelligence — used by Part 1 for OCR (Layout API)
document_intelligence_client: DocumentAnalysisClient = DocumentAnalysisClient(
    endpoint=_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(_DOC_INTEL_KEY),
)
