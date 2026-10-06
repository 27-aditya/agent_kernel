from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

VALID_TOOLS = ("run_bash", "read_file", "write_file", "git_checkout", "git_commit", "unauthorized_tool")

Verdict = Literal["ALLOW", "ESCALATE", "BLOCK"]


class ProposedAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    rationale: str
    current_task: str
    execution_history: list[str]


class DecisionResult(BaseModel):
    verdict: Verdict
    selected_tool: str
    risk_score: float
    confidence: float
    reason: str


class JevChoiceOutput(BaseModel):
    selected_option: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("selected_option")
    @classmethod
    def _known_tool(cls, value: str) -> str:
        if value not in VALID_TOOLS:
            raise ValueError(f"unknown tool {value!r}; expected one of {', '.join(VALID_TOOLS)}")
        return value


class JevNoulOutput(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str


class JevScoreOutput(BaseModel):
    score: float = Field(ge=1.0, le=5.0)
    reasoning: str


class JevParallelResponse(BaseModel):
    valid_tool: JevChoiceOutput
    is_destructive: JevNoulOutput
    blast_radius: JevScoreOutput
