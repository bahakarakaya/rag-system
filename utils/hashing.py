"""Utility functions for computing content hashes."""

import hashlib


def compute_content_hash(content: str) -> str:
    """Compute a SHA-256 hash of the given text content.

    Args:
        content: The text content to hash.

    Returns:
        A hex string representing the SHA-256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
