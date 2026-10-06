from dataclasses import dataclass
from typing import Optional

from agent_kernel.models import VALID_TOOLS, JevParallelResponse, ProposedAction, Verdict

__all__ = ["VALID_TOOLS", "PolicyThresholds", "PolicyGater"]


@dataclass(frozen=True)
class PolicyThresholds:
    block_destructive_prob: float = 0.80
    escalate_blast_radius: float = 3.5
    min_tool_confidence: float = 0.75
    uncertain_prob_floor: float = 0.40
    allow_min_confidence: float = 0.80


class PolicyGater:
    def __init__(self, thresholds: Optional[PolicyThresholds] = None) -> None:
        self.thresholds = thresholds or PolicyThresholds()

    def evaluate(
        self,
        response: JevParallelResponse,
        action: Optional[ProposedAction] = None,
    ) -> tuple[Verdict, str]:
        return self._gate(
            valid_tool_conf=response.valid_tool.confidence,
            selected_tool=response.valid_tool.selected_option,
            is_destructive_prob=response.is_destructive.probability,
            blast_radius=response.blast_radius.score,
            proposed_tool=action.tool_name if action else None,
        )

    def _gate(
        self,
        valid_tool_conf: float,
        selected_tool: str,
        is_destructive_prob: float,
        blast_radius: float,
        proposed_tool: Optional[str] = None,
    ) -> tuple[Verdict, str]:
        t = self.thresholds

        if is_destructive_prob >= t.block_destructive_prob:
            return "BLOCK", (
                f"BLOCKED: destructive probability {is_destructive_prob:.2f} >= {t.block_destructive_prob:.2f}. "
                f"Risk of permanent data loss, credential leak, or irreversible modification."
            )

        if selected_tool == "unauthorized_tool":
            return "ESCALATE", (
                "ESCALATED: judge classified the proposed tool as unauthorized_tool — "
                "requires human review."
            )

        if proposed_tool is not None and selected_tool != proposed_tool:
            return "ESCALATE", (
                f"ESCALATED: judge selected {selected_tool!r} but the action proposed "
                f"{proposed_tool!r} — requires human review."
            )

        if (
            blast_radius >= t.escalate_blast_radius
            or valid_tool_conf < t.min_tool_confidence
            or t.uncertain_prob_floor <= is_destructive_prob < t.block_destructive_prob
        ):
            reasons = []
            if blast_radius >= t.escalate_blast_radius:
                reasons.append(f"blast_radius {blast_radius:.2f} >= {t.escalate_blast_radius:.2f}")
            if valid_tool_conf < t.min_tool_confidence:
                reasons.append(
                    f"valid_tool confidence {valid_tool_conf:.2f} < {t.min_tool_confidence:.2f}"
                )
            if t.uncertain_prob_floor <= is_destructive_prob < t.block_destructive_prob:
                reasons.append(
                    f"is_destructive {is_destructive_prob:.2f} in uncertain zone "
                    f"[{t.uncertain_prob_floor:.2f}, {t.block_destructive_prob:.2f})"
                )
            return "ESCALATE", "ESCALATED: " + "; ".join(reasons) + " — requires human review."

        if is_destructive_prob < t.uncertain_prob_floor and valid_tool_conf >= t.allow_min_confidence:
            return "ALLOW", (
                f"ALLOWED: destructive probability {is_destructive_prob:.2f} < "
                f"{t.uncertain_prob_floor:.2f} and valid_tool confidence {valid_tool_conf:.2f} >= "
                f"{t.allow_min_confidence:.2f}."
            )

        return "ESCALATE", "ESCALATED: default catch-all — insufficient confidence to auto-approve."
