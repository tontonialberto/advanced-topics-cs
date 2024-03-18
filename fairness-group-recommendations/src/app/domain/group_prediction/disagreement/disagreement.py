from abc import ABC, abstractmethod

from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group


class Disagreement(ABC):
    @abstractmethod
    def get_disagreement(self, group: Group, item: ItemId) -> float:
        pass