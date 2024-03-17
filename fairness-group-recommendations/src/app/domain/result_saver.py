from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class ResultSaver(ABC):
    @abstractmethod
    def save(self, output_path: Path, headers: List[str], rows: List[List[str]]) -> None:
        pass