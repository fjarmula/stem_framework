from typing import List, Optional
from pydantic import BaseModel, Field


class CapabilityModel(BaseModel):
    """
    Represents a specific tool or skill the agent possesses.
    """
    name: str
    description: str
    parameters: str = Field(
        default="",
        description="JSON string containing parameter definitions for the capability"
    )
    required_context: List[str] = Field(default_factory=list)


class AgentGenome(BaseModel):
    """
    The complete 'DNA' of the agent at a specific version
    """
    model_config = {"extra": "forbid"}
    version: int = 1
    persona_name: str = "StemCell"
    role_description: str = "General purpose base agent."
    reasoning_protocol: str = "Zero-shot chain of thought."
    capabilities: List[CapabilityModel] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

    def rename(self, new_name: str):
        """Updates the persona name of the agent."""
        self.persona_name = new_name

    def update_role(self, new_role: str):
        """Updates the role description of the agent."""
        self.role_description = new_role


class TransformationPlan(BaseModel):
    model_config = {"extra": "forbid"}

    reasoning: str
    new_persona_name: Optional[str] = Field(
        default=None,
        description="A more specific name for the agent reflecting its new specialization"
    )
    new_role_description: Optional[str] = Field(
        default=None,
        description="A more specific role description for the agent reflecting its new specialization"
    )
    added_capabilities: List[CapabilityModel]
    removed_capabilities: List[str]
    modified_protocol: Optional[str]
    risk_assessment: str
