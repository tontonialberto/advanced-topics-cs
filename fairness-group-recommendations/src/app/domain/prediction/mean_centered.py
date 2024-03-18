from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.prediction.prediction import Prediction
from app.domain.similarity.similarity import Similarity


ALL_NEIGHBORS = -1

class MeanCenteredPrediction(Prediction):
    def __init__(self, dataset: Dataset, similarity: Similarity, num_neighbors: int, use_absolute_value: bool) -> None:
        self.__dataset = dataset
        self.__similarity = similarity
        self.__num_neighbors = num_neighbors
        self.__use_absolute_value = use_absolute_value
        self.__predictions: Dict[Tuple[UserId, ItemId], float] = {}
    
    def get_prediction(self, user: UserId, item: ItemId) -> float:
        if self.__predictions.get((user, item)):
            return self.__predictions[(user, item)]
        
        closest_neighbors = self.__get_neighbors_similarities(user, item)
        
        neighbors_avg_ratings = [
            self.__dataset.get_average_rating_by_user(neighbor)
            for (neighbor, _, _) in closest_neighbors
        ]
        rating_differences = [
            rating - avg_rating
            for ((_, rating, _), avg_rating) in zip(closest_neighbors, neighbors_avg_ratings)
        ]
        
        avg_user_rating = self.__dataset.get_average_rating_by_user(user)
        
        if self.__use_absolute_value:
            denominator = sum([abs(similarity) for (_, _, similarity) in closest_neighbors])
        else:
            denominator = sum([similarity for (_, _, similarity) in closest_neighbors])
        
        if denominator == 0:
            # Here, the case is that the considered neighbors for the given item
            # have no similarity with the considered user (ie. their tastes do not overlap in any way).
            # Just take into account the average rating for the considered user.
            prediction = avg_user_rating
        else:
            prediction = avg_user_rating + (
                sum([
                    similarity * rating_difference
                    for ((_, _, similarity), rating_difference) in zip(closest_neighbors, rating_differences)
                ])
                
                /
                
                denominator
            )
            
        self.__predictions[(user, item)] = prediction
                
        return prediction
    
    def __get_neighbors_similarities(self, user: UserId, item: ItemId) -> List[Tuple[UserId, float, float]]:
        if self.__num_neighbors == ALL_NEIGHBORS:
            # if all neighbors, return all users who rated the item
            neighbors = self.__dataset.get_users_who_rated(item)
            #   compute similarities for all users
            neighbors_similarities = [
                (
                    neighbor,
                    rating,
                    self.__similarity.get_similarity(user, neighbor)
                )
                for (neighbor, rating) in neighbors
            ]
            return neighbors_similarities
        else:
            neighbors = [neighbor for neighbor in self.__dataset.get_all_users() if neighbor != user]
            
            # compute similarities for all users
            neighbors_similarities = [
                (
                    neighbor, 
                    self.__dataset.get_rating(neighbor, item), 
                    self.__similarity.get_similarity(user, neighbor)
                )
                for neighbor in neighbors
            ]
            
            # take the most similar users
            neighbors_similarities.sort(key=lambda x: x[2], reverse=True)
            closest_neighbors = neighbors_similarities[:self.__num_neighbors]
            
            # exclude neighbors who have not rated the item
            closest_neighbors = [
                (neighbor, rating, similarity)
                for (neighbor, rating, similarity) in closest_neighbors
                if rating != 0
            ]
            
            # return the most similar users who rated the item
            return closest_neighbors