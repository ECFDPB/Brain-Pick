from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Tag:
    name: str

    def asdict(self):
        return asdict(self)


@dataclass
class Element:
    id: int
    x: float
    y: float
    width: float
    height: float
    tags: List[Tag]

    @classmethod
    def from_dict(cls, data: dict):
        tags_data = data.pop("tags", [])
        tags_objs = [Tag(**t) for t in tags_data]
        return cls(tags=tags_objs, **data)

    def asdict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tags": [t.asdict() for t in self.tags],
        }
