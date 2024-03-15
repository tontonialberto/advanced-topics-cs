from math import exp, sqrt
from app.domain.dataset import Dataset, UserId
from app.domain.similarity.similarity import Similarity


class ITR(Similarity):
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset
    
    def get_similarity(self, user_a: UserId, user_b: UserId) -> float:
        union_items = self.__dataset.get_items_rated_by_any(user_a, user_b)
        if len(union_items) == 0:
            return 0
        
        # sim triangle
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
        
        # user rating preferences similarity
        len_union_items = len(union_items)
        avg_rating_a = sum([
            rating for (rating, _) in union_items.values()
        ]) / len_union_items
        avg_rating_b = sum([
            rating for (_, rating) in union_items.values()
        ]) / len_union_items
        items_rated_by_a = self.__dataset.get_ratings_by_user(user_a)
        std_variance_a = sqrt(
            sum([
                (rating - avg_rating_a) ** 2
                for (_, rating) in items_rated_by_a
            ]) / len(items_rated_by_a)
        )
        items_rated_by_b = self.__dataset.get_ratings_by_user(user_b)
        std_variance_b = sqrt(
            sum([
                (rating - avg_rating_b) ** 2
                for (_, rating) in items_rated_by_b
            ]) / len(items_rated_by_b)
        )
        exp_argument = -1 * abs(avg_rating_a - avg_rating_b) * abs(std_variance_a - std_variance_b)
        sim_urp = 1 - (1 / (1 + exp(exp_argument)))
        
        return sim_triangle * sim_urp
        