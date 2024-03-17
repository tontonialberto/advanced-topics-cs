from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.prediction.prediction import Prediction
from app.domain.similarity.similarity import Similarity


ALL_NEIGHBORS = -1

class MeanCenteredNoAbsPrediction(Prediction):
    """
    This function is the same as the Mean Centered one,
    but the denominator does not have the absolute values of the similarities.
    """
    def __init__(self, dataset: Dataset, similarity: Similarity, num_neighbors: int) -> None:
        self.__dataset = dataset
        self.__similarity = similarity
        self.__num_neighbors = num_neighbors
    
    def get_prediction(self, user: UserId, item: ItemId) -> float:
        neighbors = [neighbor for neighbor in self.__dataset.get_all_users() if neighbor != user]
        
        # Get all users sorted by their similarity with the considered user
        neighbors_similarities = [
            (
                neighbor, 
                self.__dataset.get_rating(neighbor, item), 
                self.__similarity.get_similarity(user, neighbor)
            )
            for neighbor in neighbors
        ]
        neighbors_similarities.sort(key=lambda x: x[2], reverse=True)
        
        if self.__num_neighbors != ALL_NEIGHBORS:
            # Take only the most similar neighbors
            # Do not consider the neighbors that have not rated the item
            closest_neighbors = [
                (neighbor, rating, similarity)
                for (neighbor, rating, similarity) in  neighbors_similarities[:self.__num_neighbors]
                if rating != 0
            ]
        else:
            closest_neighbors = neighbors_similarities
        
        neighbors_avg_ratings = [
            self.__dataset.get_average_rating_by_user(neighbor)
            for (neighbor, _, _) in closest_neighbors
        ]
        rating_differences = [
            rating - avg_rating
            for ((_, rating, _), avg_rating) in zip(closest_neighbors, neighbors_avg_ratings)
        ]
        
        avg_user_rating = self.__dataset.get_average_rating_by_user(user)
        
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
                
        return prediction