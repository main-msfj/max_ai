

class MemoryError(Exception):
    """Raised for memory registry errors."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Memory error")

    @classmethod
    def missing(cls, field: str):
        return cls(f"Missing required memory field: {field}")

    @classmethod
    def unbound(cls):
        return cls(
            "Memory is not bound to a user/session: call bind(user_id, session_id) "
            "or pass it to an Agent, which binds it to each run's RunContext."
        )

    @classmethod
    def invalid_type(cls, field: str, expected: str, actual: str):
        return cls(f"Memory field {field} expected {expected}, got {actual}")


class PersistedMemoryError(MemoryError):
    """More specific memory errors for persisted stores."""

    @classmethod
    def wrong_input(cls):
        return cls("Persisted memory input is invalid.")
