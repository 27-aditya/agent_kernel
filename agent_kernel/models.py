from typing import Literal

from pydantic import BaseModel


class ProposedAction(BaseModel):
    tool_name: str
    arguments: dict
    rationale: str
    current_task: str
    execution_history: list[str]


class DecisionResult(BaseModel):
    verdict: Literal["ALLOW", "ESCALATE", "BLOCK"]
    selected_tool: str
    risk_score: float
    confidence: float
    reason: str


class JevChoiceOutput(BaseModel):
    selected_option: str
    confidence: float
    reasoning: str


class JevNoulOutput(BaseModel):
    probability: float
    reasoning: str


class JevScoreOutput(BaseModel):
    score: float
    reasoning: str


class JevParallelResponse(BaseModel):
    valid_tool: JevChoiceOutput
    is_destructive: JevNoulOutput
    blast_radius: JevScoreOutput