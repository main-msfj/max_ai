import typing as t


# -------- CHAT HISTORY -----------------------------------------------------------
class ChatHistoryError(Exception):
    def __init__(self, message: str | None):
        self.message = message
        super().__init__(message)

    @classmethod
    def wrong_input(cls, index: int):
        return cls(
            f"message_history[{index}] is a raw dict. "
            "Use ChatHistory.load_from() instead."
        )

    @classmethod
    def wrong_type(cls, index: int, actual_type: type):
        return cls(
            f"message_history[{index}] is {actual_type.__name__}, expected CoreMessage."
        )


# -------- SKILLS -----------------------------------------------------------
class RoutineRegistryError(Exception):
    def __init__(self, message: str | None):
        self.message = message
        super().__init__(message)

    @classmethod
    def invalid_skill_type(
        cls, index: int, error: str | None, actual_type: type | None
    ):
        if error:
            return cls(f"agent_skills[{index}]: invalid skill dict: {error}")
        return cls(
            f"agent_skills[{index}] must be RoutineBlocks or dict, "
            f"got {actual_type.__name__}"
        )

    @classmethod
    def wrong_type(cls, actual_type: type):
        return cls(f"agent_skills must be a list or dict, got {actual_type.__name__}")

    @classmethod
    def duplicated_skills(cls, block: t.Any):
        return cls(f"Duplicate skill name: {block.name!r}")

    @classmethod
    def required_skill_block(cls, skill: type):
        return cls(f"add() requires RoutineBlocks, got {skill.__name__}")


# -------- EXTERNAL -----------------------------------------------------------
class RetrievalContextError(Exception):
    def __init__(self, message: str | None):
        self.message = message
        super().__init__(message)

    @classmethod
    def wrong_input(cls, index: int):
        return cls(
            f"query_knowledge[{index}] is a raw dict. "
            "Use RetrievalContext.load_from() instead."
        )
