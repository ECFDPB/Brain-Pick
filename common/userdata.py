from dataclasses import dataclass

from page import Tag


@dataclass
class TagsReport:
    username: str
    timestamp: int
    topic: list[Tag]
    # Value will be a float number from -1.0 to 1.0, representing likeness.
    value: float
