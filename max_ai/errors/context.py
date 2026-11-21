

# -------- CHAT HISTORY -----------------------------------------------------------
class ChatHistoryError(Exception):
    def __init__(self, message: str | None):
        self.message = message
        super().__init__(message)

    @classmethod
    def wrong_type(cls, index: int, actual_type: type):
        return cls(
            f"message_history[{index}] is {actual_type.__name__}, expected CoreMessage."
        )
