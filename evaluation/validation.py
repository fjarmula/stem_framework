from pydantic import BaseModel, Field
from typing import List

class EnvironmentFeedback(BaseModel):
    """The signal from the environment after an agent attempts a task."""
    success: bool = Field(description="Did the agent successfully solve the task?")
    critique: str = Field(description="Detailed explanation of what the agent did wrong or couldn't do.")
    identified_gaps: List[str] = Field(
        description="Logical or mechanical missing capabilities (e.g., 'Inability to execute code', 'Lack of state memory')."
    )