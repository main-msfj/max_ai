

class StackError(Exception):
    """Error in prompt stacking system"""

    def __init__(self, message: str):
        super().__init__(message)

    @classmethod
    def template_or_load_from(cls):
        return cls(
            "Invalid configuration: use either `template` or `load_from`, not both."
        )

    @classmethod
    def missing_placeholders(cls, name: str, missing: set[str]):
        return cls(f"[{name}] Missing placeholders: {missing}")

    @classmethod
    def unexpected_placeholders(cls, name: str, unexpected: set[str]):
        return cls(f"[{name}] Unexpected placeholders: {unexpected}")

    @classmethod
    def missing_variable(cls, name: str, var: str):
        return cls(f"[{name}] Missing required variable: '{var}'")
