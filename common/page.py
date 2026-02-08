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

    @classmethod
    def from_dict(cls, data: dict):
        tags_data = data.pop("tags", [])
        tags_objs = [Tag(**t) for t in tags_data]
        return cls(tags=tags_objs, **data)
