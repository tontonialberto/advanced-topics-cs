from abc import ABC, abstractmethod

from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group


class SequentialGroupPrediction(ABC):
    @abstractmethod
    def get_prediction(self, group: Group, item: ItemId) -> float:
        pass