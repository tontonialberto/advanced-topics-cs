from abc import ABC, abstractmethod
from typing import List

from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group


class SequentialGroupRecommender(ABC):
    @abstractmethod
    def get_previous_recommendations(self, group: Group) -> List[List[ItemId]]:
        pass