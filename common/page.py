from dataclasses import dataclass


@dataclass
class Tag:
    name: str


@dataclass
class Element:
    id: int
    x: float
    y: float
    width: float
    height: float
    tags: list[Tag]
