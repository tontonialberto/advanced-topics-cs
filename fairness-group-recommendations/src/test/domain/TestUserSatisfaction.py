from typing import List, Tuple
from unittest import TestCase
from unittest.mock import ANY, Mock

from app.domain.dataset import Dataset, ItemId
from app.domain.prediction.prediction import Prediction
from app.domain.recommender import Recommender
from app.domain.user_satisfaction import UserSatisfaction


class TestUserSatisfaction(TestCase):
    def test_get_satisfaction_maximum_if_grouprecs_equal_to_userrecs(self):
        items_user_recommendation: List[ItemId] = [42, 1, 3, 100]
        ratings_user_recommendation: List[float] = [5, 4.5, 4.3, 4.2] 
        user_recommendation = list(
            zip(items_user_recommendation, ratings_user_recommendation)
        )
        group_recommendation: List[ItemId] = items_user_recommendation
        # Suppose the user has not rated any of the items in the group recommendation.
        user_actual_ratings: List[float] = [0, 0, 0, 0]
        
        user_recommender = Mock(spec=Recommender)
        user_recommender.get_recommendations.return_value = user_recommendation
        user_predictor = Mock(spec=Prediction)
        user_predictor.get_prediction.side_effect = ratings_user_recommendation
        dataset = Mock(spec=Dataset)
        dataset.get_rating.side_effect = user_actual_ratings
        satisfaction = UserSatisfaction(user_recommender, user_predictor, dataset)
        
        sat = satisfaction.get_satisfaction(1, group_recommendation)
        
        self.assertEqual(1, sat)
        
    def test_get_satisfaction_uses_prediction_on_grouprecs_if_user_has_not_rated_items(self) -> None:
        items_user_recommendation: List[ItemId] = [7, 81, 666, 12]
        ratings_user_recommendation: List[float] = [5, 5, 5, 5]
        user_recommendation = list(zip(items_user_recommendation, ratings_user_recommendation))
        group_recommendation: List[ItemId] = [42, 1, 3, 100]
        # Suppose the user has not rated any of the items in the group recommendation.
        user_actual_ratings: List[float] = [0, 0, 0, 0]
        # Also suppose that the user doesn't like the items in the group recommendation.
        user_predicted_ratings: List[float] = [1, 1, 1, 1]
        
        user_recommender = Mock(spec=Recommender)
        user_recommender.get_recommendations.return_value = user_recommendation
        user_predictor = Mock(spec=Prediction)
        user_predictor.get_prediction.side_effect = user_predicted_ratings
        dataset = Mock(spec=Dataset)
        dataset.get_rating.side_effect = user_actual_ratings
        satisfaction = UserSatisfaction(user_recommender, user_predictor, dataset)
        
        sat = satisfaction.get_satisfaction(1, group_recommendation)
        
        self.assertEqual(4 / 20, sat)
        
    def test_get_satisfaction_uses_dataset_on_grouprecs_if_user_has_rated_items(self) -> None:
        items_user_recommendation: List[ItemId] = [7, 81, 666, 12]
        ratings_user_recommendation: List[float] = [5, 5, 5, 5]
        user_recommendation = list(zip(items_user_recommendation, ratings_user_recommendation))
        group_recommendation: List[ItemId] = [42, 1, 3, 100]
        # Suppose that the user has rated every item in the group recommendation.
        user_actual_ratings: List[float] = [1, 1, 1, 1]
        
        user_recommender = Mock(spec=Recommender)
        user_recommender.get_recommendations.return_value = user_recommendation
        user_predictor = Mock(spec=Prediction)
        dataset = Mock(spec=Dataset)
        dataset.get_rating.side_effect = user_actual_ratings
        satisfaction = UserSatisfaction(user_recommender, user_predictor, dataset)
        
        sat = satisfaction.get_satisfaction(1, group_recommendation)
        
        self.assertEqual(4 / 20, sat)