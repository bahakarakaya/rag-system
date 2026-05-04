"""Helpers for optional Langfuse tracing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from langfuse import Langfuse, get_client

_LANGFUSE_CLIENT: Langfuse | None = None


def get_langfuse_client() -> Langfuse | None:
    """Return a cached Langfuse client when configured, otherwise None."""
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    host = os.getenv("LANGFUSE_HOST")
    base_url = os.getenv("LANGFUSE_BASE_URL")
    if host and not base_url:
        os.environ["LANGFUSE_BASE_URL"] = host

    _LANGFUSE_CLIENT = get_client()
    return _LANGFUSE_CLIENT


@contextmanager
def start_observation(
    langfuse: Langfuse | None,
    name: str,
    as_type: str = "span",
    **kwargs: object,
) -> Generator[object | None, None, None]:
    """Yield a Langfuse observation context when enabled, otherwise None."""
    if langfuse is None:
        yield None
        return

    with langfuse.start_as_current_observation(
        as_type=as_type,
        name=name,
        **kwargs,
    ) as observation:
        yield observation
