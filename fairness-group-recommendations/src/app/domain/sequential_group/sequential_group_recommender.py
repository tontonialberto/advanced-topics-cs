from abc import ABC, abstractmethod
from typing import List, Tuple

from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group


class SequentialGroupRecommender(ABC):
    @abstractmethod
    def get_previous_recommendations(self, group: Group) -> List[List[ItemId]]:
        pass
    
    @abstractmethod
    def get_recommendations(self, group: Group, limit: int) -> List[Tuple[ItemId, float]]:
        pass