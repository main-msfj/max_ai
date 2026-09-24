"""Shared short-id generation for records, events and sessions.

8 hex chars = 32 bits of entropy — plenty for ids scoped to one run,
one tool-state dict, or one user's sessions; not meant for anything
compared across a large shared namespace.
"""

import uuid


def short_id() -> str:
    return uuid.uuid4().hex[:8]
