"""
LLM client — wraps IBM watsonx (primary) with OpenAI fallback.
Returns a LangChain-compatible model so all agents stay LLM-agnostic.

Response helper:  use extract_text(response) to get a plain string
from either a WatsonxLLM (returns str) or ChatOpenAI (returns AIMessage).
"""
from __future__ import annotations
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


def extract_text(response) -> str:
    """
    Normalize LLM response to a plain string.
    - WatsonxLLM  → already a str
    - ChatOpenAI  → AIMessage with a .content attribute
    """
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def get_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """
    Return an LLM instance.

    Priority:
      1. IBM watsonx Granite
      2. OpenAI gpt-4o-mini
      3. RuntimeError if neither is configured
    """
    if settings.has_watsonx:
        try:
            from langchain_ibm import WatsonxLLM

            llm = WatsonxLLM(
                model_id="ibm/granite-13b-instruct-v2",
                url=settings.watsonx_url,
                apikey=settings.watsonx_api_key,
                project_id=settings.watsonx_project_id,
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": 1,
                    "temperature": temperature,
                    "top_k": 50,
                    "top_p": 1,
                },
            )
            logger.info("Using IBM watsonx Granite LLM")
            return llm
        except Exception as exc:
            logger.warning("watsonx init failed: %s — falling back to OpenAI", exc)

    if settings.has_openai:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("Using OpenAI gpt-4o-mini")
            return llm
        except Exception as exc:
            logger.warning("OpenAI init failed: %s", exc)

    raise RuntimeError(
        "No LLM configured.\n"
        "Set OPENAI_API_KEY in your .env file  —OR—  "
        "set WATSONX_API_KEY + WATSONX_PROJECT_ID for IBM watsonx."
    )


def get_embeddings():
    """
    HuggingFace sentence-transformers embeddings — runs fully locally,
    no API key needed.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Using HuggingFace all-MiniLM-L6-v2 embeddings")
        return embeddings
    except Exception as exc:
        logger.error("Failed to load embeddings model: %s", exc)
        raise
