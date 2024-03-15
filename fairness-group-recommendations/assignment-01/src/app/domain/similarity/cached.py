from typing import Dict, Tuple
from app.domain.dataset import ItemId, UserId
from app.domain.similarity.similarity import Similarity


class CachedSimilarity(Similarity):
    def __init__(self, similarity: Similarity) -> None:
        super().__init__()
        self.__similarities: Dict[Tuple[UserId, UserId], float] = {}
        self.__similarity = similarity
        
    def get_similarity(self, user_a: ItemId, user_b: ItemId) -> float:
        similarity = self.__similarities.get((user_a, user_b))
        if similarity is not None:
            return similarity
        
        similarity = self.__similarities.get((user_b, user_a))
        if similarity is not None:
            return similarity
        
        similarity = self.__similarity.get_similarity(user_a, user_b)
        self.__similarities.update({(user_a, user_b): similarity})
        
        return similarity