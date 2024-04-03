from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, UserId
from app.domain.similarity.similarity import Similarity


class Stats:
    def __init__(self, dataset: Dataset, similarity: Similarity) -> None:
        self.__dataset = dataset
        self.__similarity = similarity
    
    def get_most_similar_users(self, user: UserId, limit: int) -> List[Tuple[UserId, float]]:
        all_other_users = [user_id for user_id in self.__dataset.get_all_users() if user_id != user]
        
        user_similarities: List[Tuple[UserId, float]] = [
            (neighbor, self.__similarity.get_similarity(user, neighbor))
            for neighbor in all_other_users
        ]
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return user_similarities[:limit]
    
    def get_user_similarity_matrix(self) -> Dict[Tuple[UserId, UserId], float]:
        all_users = self.__dataset.get_all_users()
        user_similarities = {}
        for user_a in all_users:
            for user_b in all_users:
                user_similarities[(user_a, user_b)] = self.__similarity.get_similarity(user_a, user_b)
        return user_similarities