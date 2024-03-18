from abc import ABC, abstractmethod
from typing import List

from app.domain.dataset import ItemId, UserId

Group = List[UserId]

class GroupPrediction(ABC):
    @abstractmethod
    def get_prediction(self, group: Group, item: ItemId) -> float:
        pass