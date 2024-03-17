from abc import ABC, abstractmethod

from app.domain.dataset import UserId


class Similarity(ABC):
    @abstractmethod
    def get_similarity(self, user_a: UserId, user_b: UserId) -> float:
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__