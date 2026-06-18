from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReviewMedia:
    label: str
    path: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewItem:
    case_id: str
    title: str
    rank: int
    part_number: str
    file_name: str
    view: str
    risk_score: float
    risk_level: str
    risk_reasons: list[str]
    media: list[ReviewMedia]
    links: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["media"] = [item.to_dict() for item in self.media]
        return payload

