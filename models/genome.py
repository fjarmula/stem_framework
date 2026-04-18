from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CapabilityModel(BaseModel):
    """
    Represents a specific tool or skill the agent possesses.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_context: List[str] = Field(
        description="Data points needed for the environment to execute the capability",
        default_factory=list
    )

class AgentGenome(BaseModel):
    """The complete 'DNA' of the agent at a specific version"""
    version: int = 1
    persona_name: str ="StemCell"
    role_description: str = "General purpose base agent."
    reasoning_protocol: str = "Zero-shot chain of thought."
    capabilities: List[CapabilityModel] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

class TransformationPlan(BaseModel):
    """A proposed mutation of the AgentGenome"""
    reasoning: str = Field(description="Why is this transformation necessary?")
    added_capabilities: List[CapabilityModel]
    removed_capabilities: List[str]
    modified_protocol: Optional[str]
    risk_assessment: str