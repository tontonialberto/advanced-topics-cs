from math import exp, sqrt
from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.similarity.similarity import Similarity


class ITR(Similarity):
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset
    
    def get_similarity(self, user_a: UserId, user_b: UserId) -> float:
        union_items = self.__dataset.get_items_rated_by_any(user_a, user_b)
        if len(union_items) == 0:
            return 0
        
        sim_triangle = self.get_similarity_triangle_improved(union_items)
        
        sim_urp = self.get_similarity_urp(union_items)
                
        return sim_triangle * sim_urp
    
    def get_similarity_triangle_improved(self, union_items: Dict[ItemId, Tuple[float, float]]) -> float:
        rating_differences_squared = [
            (rating_a - rating_b) ** 2
            for (rating_a, rating_b) in union_items.values()
        ]
        ratings_a_squared = [
            rating ** 2
            for (rating, _) in union_items.values()
        ]
        ratings_b_squared = [
            rating ** 2
            for (_, rating) in union_items.values()
        ]
        numerator = sqrt(sum(rating_differences_squared))
        denominator = sqrt(sum(ratings_a_squared)) + sqrt(sum(ratings_b_squared))
        sim_triangle = 1 - (numerator / denominator)
        return sim_triangle
    
    def get_average_and_stddev(self, ratings: List[float], num_of_union_items: int) -> Tuple[float, float]:
        avg_rating = sum(ratings) / num_of_union_items
        numerator_std_variance = sum([
            (rating - avg_rating)
            for rating in ratings
        ]) ** 2
        std_variance = sqrt(numerator_std_variance / len(ratings))
        return (avg_rating, std_variance)
    
    def get_similarity_urp(self, union_items: Dict[ItemId, Tuple[float, float]]) -> float:
        items_rated_by_a = [rating for (rating, _) in union_items.values() if rating != 0]
        avg_rating_a, std_variance_a = self.get_average_and_stddev(
            items_rated_by_a,
            len(union_items)
        )
        
        items_rated_by_b = [rating for (_, rating) in union_items.values() if rating != 0]
        avg_rating_b, std_variance_b = self.get_average_and_stddev(
            items_rated_by_b,
            len(union_items)
        )
        
        argument = -1 * abs(avg_rating_a - avg_rating_b) * abs(std_variance_a - std_variance_b)
        sim_urp = 1 - (1 / (1 + exp(argument)))
        return sim_urp
        