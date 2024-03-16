from abc import ABC, abstractmethod

from app.domain.dataset import ItemId, UserId


class Prediction(ABC):
    @abstractmethod
    def get_prediction(self, user: UserId, item: ItemId) -> float:
        pass