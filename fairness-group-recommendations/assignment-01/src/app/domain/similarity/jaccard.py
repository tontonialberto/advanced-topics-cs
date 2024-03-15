from app.domain.dataset import Dataset, UserId
from app.domain.similarity.similarity import Similarity


class Jaccard(Similarity):
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset
    
    def get_similarity(self, user_a: UserId, user_b: UserId) -> float:
        union_items = self.__dataset.get_items_rated_by_any(user_a, user_b)
        
        if len(union_items) == 0:
            return 0
        
        common_items = self.__dataset.get_items_rated_by_both(user_a, user_b)
        return len(common_items) / len(union_items)