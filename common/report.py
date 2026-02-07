from dataclasses import dataclass

from common.page import Tag


@dataclass
class UserReport:
    username: str
    values: dict[Tag, float]


@dataclass
class ProtectedReport:
    values: dict[Tag, float]
