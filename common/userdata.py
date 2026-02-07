from dataclasses import dataclass


@dataclass
class Tag:
    name: str


@dataclass
class TagsReport:
    username: str
    timestamp: int
    topic: list[Tag]
    # Value will be a float number from -1.0 to 1.0, representing likeness.
    value: float
