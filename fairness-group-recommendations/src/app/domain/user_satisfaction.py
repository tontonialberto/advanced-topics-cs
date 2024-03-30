from typing import List
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.prediction.prediction import Prediction
from app.domain.recommender import Recommender


class UserSatisfaction:
    def __init__(self, recommender: Recommender, prediction: Prediction, dataset: Dataset) -> None:
        self.__recommender = recommender
        self.__prediction = prediction
        self.__dataset = dataset
    
    def get_satisfaction(self, user: UserId, group_recommendation: List[ItemId]) -> float:
        # For each item in the group recommendation, get the rating of the user.
        # If the user has not rated the item, use the prediction to get the rating.
        group_list_user_ratings = []
        for item in group_recommendation:
            rating = self.__dataset.get_rating(user, item)
            if rating == 0:
                rating = self.__prediction.get_prediction(user, item)
            group_list_user_ratings.append(rating)
        
        group_list_satisfaction = sum(group_list_user_ratings)
        
        user_recommendations = self.__recommender.get_recommendations(user, len(group_recommendation))
        user_list_satisfaction = sum([rating for _, rating in user_recommendations])
        
        return group_list_satisfaction / user_list_satisfaction