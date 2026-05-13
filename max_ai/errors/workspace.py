from pathlib import Path

class WorkSpaceError(Exception):
    """Raised for workspace registry errors."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "Workspace error")

    @classmethod
    def no_file_exist(cls, source: Path):
        return FileNotFoundError(f"File does not exist: {source}")
    
    @classmethod
    def out_of_workspace(cls):
        return ValueError("Filename must stay inside workspace directory")
    
    @classmethod
    def no_workspace_file_exist(cls, filename: str):
        return FileNotFoundError(f"Workspace file does not exist: {filename}")
    