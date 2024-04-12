from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    classes: int = 301
    pretrained_path: Optional[str] = None
