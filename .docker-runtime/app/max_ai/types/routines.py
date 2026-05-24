from pydantic import BaseModel, Field


# -------- ROUTINES -----------------------------------------------------------
class RoutineSummary(BaseModel):
    name: str = Field(..., description="Routine identifier")
    description: str = Field(..., description="Short description")
