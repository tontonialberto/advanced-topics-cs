from abc import ABC, abstractmethod

from app.domain.dataset import Dataset


class DataLoader(ABC):
    @abstractmethod
    def load() -> Dataset:
        pass