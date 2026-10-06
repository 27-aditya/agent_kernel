from typing import Literal

from agent_kernel.models import JevParallelResponse


VALID_TOOLS = ["run_bash", "read_file", "write_file", "git_checkout", "git_commit", "unauthorized_tool"]


class PolicyGater:
    def evaluate(self, response: JevParallelResponse) -> tuple[str, str]:
        verdict, reason = self._gate(
            valid_tool_conf=response.valid_tool.confidence,
            selected_tool=response.valid_tool.selected_option,
            is_destructive_prob=response.is_destructive.probability,
            blast_radius=response.blast_radius.score,
        )
        return verdict, reason

    def _gate(
        self,
        valid_tool_conf: float,
        selected_tool: str,
        is_destructive_prob: float,
        blast_radius: float,
    ) -> tuple[str, str]:
        if is_destructive_prob >= 0.80:
            return "BLOCK", (
                f"BLOCKED: destructive probability {is_destructive_prob:.2f} >= 0.80. "
                f"Risk of permanent data loss, credential leak, or irreversible modification."
            )

        if blast_radius >= 3.5 or valid_tool_conf < 0.75 or 0.40 <= is_destructive_prob < 0.80:
            reasons = []
            if blast_radius >= 3.5:
                reasons.append(f"blast_radius {blast_radius:.2f} >= 3.5")
            if valid_tool_conf < 0.75:
                reasons.append(f"valid_tool confidence {valid_tool_conf:.2f} < 0.75")
            if 0.40 <= is_destructive_prob < 0.80:
                reasons.append(f"is_destructive {is_destructive_prob:.2f} in uncertain zone [0.40, 0.80)")
            return "ESCALATE", "ESCALATED: " + "; ".join(reasons) + " — requires human review."

        if is_destructive_prob < 0.40 and valid_tool_conf >= 0.80:
            return "ALLOW", (
                f"ALLOWED: destructive probability {is_destructive_prob:.2f} < 0.40 "
                f"and valid_tool confidence {valid_tool_conf:.2f} >= 0.80."
            )

        return "ESCALATE", "ESCALATED: default catch-all — insufficient confidence to auto-approve."
