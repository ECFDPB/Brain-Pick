from dataclasses import dataclass


@dataclass
class Tag:
    name: str


@dataclass
class Element:
    id: int
    width: float
    height: float
    tags: list[Tag]
