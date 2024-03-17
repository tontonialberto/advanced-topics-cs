from typing import List
from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction import Group, GroupPrediction
from app.domain.prediction.prediction import Prediction


class AverageAggregation(GroupPrediction):
    def __init__(self, dataset: Dataset, user_predictor: Prediction) -> None:
        self.__dataset = dataset
        self.__user_predictor = user_predictor
    
    def get_prediction(self, group: Group, item: ItemId) -> float:
        user_ratings: List[float] = []
        for user in group:
            user_rating = self.__dataset.get_rating(user, item)
            if user_rating == 0: # User has not rated the item. Predict the value
                user_rating = self.__user_predictor.get_prediction(user, item)
            user_ratings.append(user_rating)
        
        return sum(user_ratings) / len(group)