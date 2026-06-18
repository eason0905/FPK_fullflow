from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    stage: str
    task: str
    index: int
    image_path: str
    image_rel_path: str
    expected: str
    predicted: str
    reason: str
    source_path: str
    prompt: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.+／-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "item"


def relative_path(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()
