from dataclasses import dataclass, asdict
from typing import List

from common.page import Tag


@dataclass
class UserReport:
    username: str
    timestamp: int
    topic: List[str]
    value: float  # -1.0 to 1.0

    def asdict(self):
        return asdict(self)


@dataclass
class ProtectedReport:
    topic: List[str]
    value: float  # -1.0 to 1.0

    def asdict(self):
        return asdict(self)
