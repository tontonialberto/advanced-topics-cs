from math import sqrt
from app.domain.dataset import Dataset, UserId
from app.domain.similarity.similarity import Similarity


class PearsonCorrelation(Similarity):
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset
    
    def get_similarity(self, user_a: UserId, user_b: UserId) -> float:
        if user_a == user_b:
            return 1
        
        avg_rating_a = self.__dataset.get_average_rating_by_user(user_a)
        avg_rating_b = self.__dataset.get_average_rating_by_user(user_b)
        
        items_rated_by_both = self.__dataset.get_items_rated_by_both(user_a, user_b)
        
        numerator = sum([
            (rating_a - avg_rating_a) * (rating_b - avg_rating_b)
            for rating_a, rating_b in items_rated_by_both.values()
        ])
        
        ratings_a = [rating for (rating, _) in items_rated_by_both.values()]
        sum_of_squares_a = sum([
            (rating - avg_rating_a) ** 2
            for rating in ratings_a
        ])
        
        ratings_b = [rating for (_, rating) in items_rated_by_both.values()]
        sum_of_squares_b = sum([
            (rating - avg_rating_b) ** 2
            for rating in ratings_b
        ])
        
        if sum_of_squares_a == 0 or sum_of_squares_b == 0:
            return 0
        
        return numerator / sqrt(sum_of_squares_a * sum_of_squares_b)