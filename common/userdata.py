from dataclasses import dataclass, asdict
from typing import List

from common.page import Tag


@dataclass
class TagsReport:
    username: str
    timestamp: int
    topic: List[Tag]
    value: float  # -1.0 to 1.0, represents likeness

    def asdict(self):
        return {
            "username": self.username,
            "timestamp": self.timestamp,
            "topic": [t.asdict() for t in self.topic],
            "value": self.value,
        }
