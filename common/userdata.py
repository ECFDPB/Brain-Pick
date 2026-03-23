from dataclasses import dataclass
from typing import List

from common.page import Tag


@dataclass
class TagsReport:
    username: str
    timestamp: int
    topic: List[Tag]
    value: float  # -1.0 to 1.0, represents likeness
