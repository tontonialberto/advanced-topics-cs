from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.similarity.similarity import Similarity

    
class Prediction(ABC):
    @abstractmethod
    def get_prediction(self, user: UserId, item: ItemId) -> float:
        pass
    
class PredictionImpl(Prediction):
    def __init__(self, dataset: Dataset, similarity: Similarity) -> None:
        self.__dataset = dataset
        self.__similarity = similarity
    
    def get_prediction(self, user: ItemId, item: ItemId) -> float:
        neighbors = self.__dataset.get_users_who_rated(item)
        neighbors_avg_ratings = [
            self.__dataset.get_average_rating_by_user(neighbor)
            for (neighbor, _) in neighbors
        ]
        rating_differences = [
            rating - avg_rating
            for ((_, rating), avg_rating) in zip(neighbors, neighbors_avg_ratings)
        ]
        
        similarities = [
            self.__similarity.get_similarity(user, neighbor)
            for (neighbor, _) in neighbors
        ]
        
        avg_user_rating = self.__dataset.get_average_rating_by_user(user)
        
        denominator = sum([abs(similarity) for similarity in similarities])
        
        if denominator == 0:
            # Here, the case is that the considered neighbors for the given item
            # have no similarity with the considered user (ie. their tastes do not overlap in any way).
            # Just take into account the average rating for the considered user.
            prediction = avg_user_rating
        else:
            prediction = avg_user_rating + (
                sum([
                    similarity * rating_difference
                    for (similarity, rating_difference) in zip(similarities, rating_differences)
                ])
                
                /
                
                denominator
            )
                
        return prediction
    
class Stats:
    def __init__(self, dataset: Dataset, similarity: Similarity) -> None:
        self.__dataset = dataset
        self.__similarity = similarity
    
    def get_most_similar_users(self, user: UserId, limit: int) -> List[Tuple[UserId, float]]:
        all_other_users = set([
            user_id 
            for (user_id, _, _) in self.__dataset.data()
            if user_id != user
        ])
        
        user_similarities = [
            (neighbor, self.__similarity.get_similarity(user, neighbor))
            for neighbor in all_other_users
        ]
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return user_similarities[:limit]
    
class Recommender: 
    def __init__(self, dataset: Dataset, predictor: Prediction) -> None:
        self.__dataset = dataset
        self.__predictor = predictor
    
    def get_recommendations(self, user: UserId, limit: int) -> List[Tuple[ItemId, float]]:
        unrated_items = self.__dataset.get_items_not_rated_by_user(user)
        predicted_ratings = [
            (item, self.__predictor.get_prediction(user, item))
            for item in unrated_items
        ]
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return predicted_ratings[:limit]

@dataclass
class PredictionComparison:
    item: ItemId
    actual_rating: float
    predicted_ratings: List[float]
    errors: List[float]

class PerformanceEvaluator:
    def __init__(self, predictors: List[Tuple[str, Prediction]], dataset: Dataset) -> None:
        self.__predictors = predictors
        self.__dataset = dataset
    
    def get_comparison_by_user(self, user: UserId) -> Dict[ItemId, PredictionComparison]:
        user_ratings = self.__dataset.get_ratings_by_user(user)
        comparison = {}
        for item, actual_rating in user_ratings:
            predicted_ratings = [
                predictor.get_prediction(user, item)
                for (_, predictor) in self.__predictors
            ]
            errors = [
                abs(actual_rating - predicted_rating)
                for predicted_rating in predicted_ratings
            ]
            comparison[item] = PredictionComparison(
                item, actual_rating, predicted_ratings, errors
            )
        return comparison
    
    @property
    def predictor_names(self) -> List[str]:
        return [name for (name, _) in self.__predictors]