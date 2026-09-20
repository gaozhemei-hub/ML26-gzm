"""Enforce and document a validation-only model selection protocol."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class EvaluationProtocolError(RuntimeError):
    """Raised when code tries to use the held-out test set prematurely."""


@dataclass
class ProtocolEvent:
    stage: str
    model_name: str
    purpose: str


@dataclass
class EvaluationProtocolGuard:
    """Small state machine that makes test-set access explicit and auditable.

    Candidate models may record validation metrics freely.  Exactly one model
    must then be locked from validation before a single test evaluation is
    authorized.  This object is framework independent and can wrap any model
    search loop.
    """

    validation_candidates: list[str] = field(default_factory=list)
    selected_model: str | None = None
    test_accesses: list[ProtocolEvent] = field(default_factory=list)
    events: list[ProtocolEvent] = field(default_factory=list)

    def record_validation(self, model_name: str, purpose: str = "candidate evaluation") -> None:
        if self.selected_model is not None:
            raise EvaluationProtocolError("模型已锁定，不能再加入新的候选模型。")
        if model_name not in self.validation_candidates:
            self.validation_candidates.append(model_name)
        self.events.append(ProtocolEvent("validation", model_name, purpose))

    def lock_validation_choice(self, model_name: str, purpose: str = "best validation metric") -> None:
        if model_name not in self.validation_candidates:
            raise EvaluationProtocolError("只能锁定已经在验证集上评估过的候选模型。")
        if self.selected_model is not None and self.selected_model != model_name:
            raise EvaluationProtocolError("验证集最佳模型已锁定，不能用其他模型替换。")
        self.selected_model = model_name
        self.events.append(ProtocolEvent("selection", model_name, purpose))

    def authorize_final_test(self, model_name: str, purpose: str = "final held-out evaluation") -> None:
        # Test labels must not influence either candidate generation or selection.
        # Keeping this guard stateful makes an accidental second access fail loudly.
        if self.selected_model is None:
            raise EvaluationProtocolError("测试集访问被拒绝：请先根据验证集锁定唯一模型。")
        if model_name != self.selected_model:
            raise EvaluationProtocolError("测试集访问被拒绝：只能评估验证集锁定的模型。")
        if self.test_accesses:
            raise EvaluationProtocolError("测试集访问被拒绝：最终评估已经完成。")
        event = ProtocolEvent("test", model_name, purpose)
        self.test_accesses.append(event)
        self.events.append(event)

    @property
    def compliant(self) -> bool:
        return self.selected_model is not None and len(self.test_accesses) == 1

    @property
    def status(self) -> str:
        if self.compliant:
            return "complete"
        if self.selected_model is not None:
            return "selection_locked"
        if self.validation_candidates:
            return "validation_in_progress"
        return "not_started"

    def to_dict(self) -> dict:
        return {
            "policy": "候选模型只根据验证集选择；测试集仅允许被最终锁定的模型访问一次。",
            "validation_candidates": self.validation_candidates,
            "selected_model": self.selected_model,
            "test_access_count": len(self.test_accesses),
            "compliant": self.compliant,
            "status": self.status,
            "events": [asdict(e) for e in self.events],
        }


def write_protocol_artifacts(guard: EvaluationProtocolGuard, output_dir: str | Path) -> Path:
    destination = Path(output_dir) / "evaluation_protocol"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "test_access_log.json"
    path.write_text(json.dumps(guard.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
