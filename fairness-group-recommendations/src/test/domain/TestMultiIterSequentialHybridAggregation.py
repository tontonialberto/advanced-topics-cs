from typing import List
from unittest import TestCase
from unittest.mock import ANY, Mock
from parameterized import parameterized

from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group, GroupPrediction
from app.domain.sequential_group.prediction.multi_iter_sequential_hybrid_aggregation import MultiIterSequentialHybridAggregation
from app.domain.sequential_group.sequential_group_recommender import SequentialGroupRecommender
from app.domain.user_satisfaction import UserSatisfaction


class TestSequentialGroupPrediction(TestCase):
    @parameterized.expand([
        (
            [ANY, ANY], # Group of users
            ANY,
            [], # No previous iterations
            1, # Consider only the last iteration to compute disagreement
            4.3, # Average prediction
            6666 # Least misery prediction, should not be used
        ),
        (
            [ANY, ANY], # Group of users
            ANY,
            [ANY, ANY, ANY, ANY], # Many previous iterations
            100, # Consider the last 100 iterations to compute disagreement (more than the number of iterations available)
            4.3, # Average prediction
            6666 # Least misery prediction, should not be used
        )
    ])
    def test_get_prediction_should_be_average_at_initial_iterations(
            self,
            group: Group,
            item: ItemId,
            previous_recommendations: List,
            iterations_to_consider: int,
            average_prediction: float,
            least_misery_prediction: float):
        
        recommender = Mock(spec=SequentialGroupRecommender)
        recommender.get_previous_recommendations.return_value = previous_recommendations
        group_predictor_avg = Mock(spec=GroupPrediction)
        group_predictor_avg.get_prediction.return_value = average_prediction
        group_predictor_least_misery = Mock(spec=GroupPrediction)
        group_predictor_least_misery.get_prediction.return_value = least_misery_prediction
        predictor = MultiIterSequentialHybridAggregation(
            recommender=recommender,
            predictor_average=group_predictor_avg,
            predictor_least_misery=group_predictor_least_misery,
            user_satisfaction=Mock(spec=UserSatisfaction),
            iterations_to_consider=iterations_to_consider,
        )
        
        self.assertEqual(average_prediction, predictor.get_prediction(group, item))
    
    @parameterized.expand([
        (
            [1, 2],
            ANY,
            [ANY], # Just one previous iteration
            1, # Consider only the last iteration to compute disagreement
            [0.5, 0.5], # All users equally satisfied => no disagreement
            4.3,
            6666 # Least misery prediction, should not be used
        ),
        (
            # Same as before, but many previous iterations are available
            [1, 2],
            ANY,
            [ANY, ANY, ANY], 
            1, # Consider only the last iteration to compute disagreement
            [0.5, 0.5],
            4.3,
            6666
        ),
        (
            [1, 2],
            ANY,
            [ANY, ANY, ANY], 
            2, # Consider last two iterations to compute disagreement
            [
                0.5, 0.5, # Satisfaction in (last-1) iteration
                0.8, 0.8, # Satisfaction in last iteration
            ],
            4.3,
            6666
        ),
    ])
    def test_get_prediction_should_be_average_when_no_disagreement(
            self,
            group: Group,
            item: ItemId,
            previous_recommendations: List,
            iterations_to_consider: int,
            user_satisfactions: List,
            average_prediction: float,
            least_misery_prediction: float):
        
        recommender = Mock(spec=SequentialGroupRecommender)
        recommender.get_previous_recommendations.return_value = previous_recommendations
        group_predictor_avg = Mock(spec=GroupPrediction)
        group_predictor_avg.get_prediction.return_value = average_prediction
        group_predictor_least_misery = Mock(spec=GroupPrediction)
        group_predictor_least_misery.get_prediction.return_value = least_misery_prediction
        user_satisfaction = Mock(spec=UserSatisfaction)
        user_satisfaction.get_satisfaction.side_effect = user_satisfactions
        predictor = MultiIterSequentialHybridAggregation(
            recommender=recommender,
            predictor_average=group_predictor_avg,
            predictor_least_misery=group_predictor_least_misery,
            user_satisfaction=user_satisfaction,
            iterations_to_consider=iterations_to_consider,
        )
        
        self.assertEqual(
            average_prediction, 
            predictor.get_prediction(group, item)
        )
        
    @parameterized.expand([
        (
            [1, 2],
            ANY,
            [ANY], # Just one previous iteration
            1, # Consider only the last iteration to compute disagreement
            [0, 1], # Maximum disagreement
            6666, # Average prediction, should not be used
            1.2 # Least misery prediction
        ),
        (
            [1, 2],
            ANY,
            [ANY, ANY, ANY], # many previous iterations are available
            1, # Consider only the last iteration to compute disagreement
            [1, 0],
            6666,
            1.2,
        ),
        (
            [1, 2],
            ANY,
            [ANY, ANY, ANY], 
            3, # Consider last 3 iterations to compute disagreement
            [
                1, 0,
                0, 1,
                0, 1,
            ],
            6666,
            1.2,
        )
    ])
    def test_get_prediction_should_be_least_misery_when_maximum_disagreement(
            self,
            group: Group,
            item: ItemId,
            previous_recommendations: List,
            iterations_to_consider: int,
            user_satisfactions: List,
            average_prediction: float,
            least_misery_prediction: float):
        
        recommender = Mock(spec=SequentialGroupRecommender)
        recommender.get_previous_recommendations.return_value = previous_recommendations
        group_predictor_avg = Mock(spec=GroupPrediction)
        group_predictor_avg.get_prediction.return_value = average_prediction
        group_predictor_least_misery = Mock(spec=GroupPrediction)
        group_predictor_least_misery.get_prediction.return_value = least_misery_prediction
        user_satisfaction = Mock(spec=UserSatisfaction)
        user_satisfaction.get_satisfaction.side_effect = user_satisfactions
        predictor = MultiIterSequentialHybridAggregation(
            recommender=recommender,
            predictor_average=group_predictor_avg,
            predictor_least_misery=group_predictor_least_misery,
            user_satisfaction=user_satisfaction,
            iterations_to_consider=iterations_to_consider,
        )
        
        self.assertEqual(
            least_misery_prediction, 
            predictor.get_prediction(group, item)
        )
    
    @parameterized.expand([
        (
            # Weighted sum of average and least misery. Weight is 0.5
            [ANY, ANY],
            ANY,
            [ANY],
            1,
            [0.5, 1], # Disagreement is 0.5
            4.2,
            1.2,
            (4.2 * 0.5) + (1.2 * 0.5)
        ),
        (
            [ANY, ANY],
            ANY,
            [ANY, ANY, ANY, ANY],
            3,
            [
                0.5, 1,
                0, 0.8, # Maximum disagreement
                0.5, 0.4,
            ],
            4.2,
            1.2,
            (4.2 * 0.2) + (1.2 * 0.8)
        ),
        (
            # Test with more than 2 users
            [ANY, ANY, ANY, ANY], # group of 4 users
            ANY,
            [ANY, ANY, ANY, ANY, ANY], # 5 previous iterations
            4, # Consider last 4 iterations to compute disagreement
            [
                1,      0.95, 0.89, 0.4, 
                0.86,   0.90, 0.78, 0.5,
                0.90,   0.98, 0.86, 0.24, # maximum disagreement = 0.98 - 0.24
                0.60,   0.70, 0.80, 0.9,
            ],
            4.2,
            1.2,
            (4.2 * 0.26) + (1.2 * 0.74)
        ),
    ])
    def test_get_prediction_should_be_weighted_sum_of_avg_and_leastmisery_when_disagreement_is_neither_max_nor_min(
            self,
            group: Group,
            item: ItemId,
            previous_recommendations: List,
            iterations_to_consider: int,
            user_satisfactions: List,
            average_prediction: float,
            least_misery_prediction: float,
            expected_prediction: float):
        
        recommender = Mock(spec=SequentialGroupRecommender)
        recommender.get_previous_recommendations.return_value = previous_recommendations
        group_predictor_avg = Mock(spec=GroupPrediction)
        group_predictor_avg.get_prediction.return_value = average_prediction
        group_predictor_least_misery = Mock(spec=GroupPrediction)
        group_predictor_least_misery.get_prediction.return_value = least_misery_prediction
        user_satisfaction = Mock(spec=UserSatisfaction)
        user_satisfaction.get_satisfaction.side_effect = user_satisfactions
        predictor = MultiIterSequentialHybridAggregation(
            recommender=recommender,
            predictor_average=group_predictor_avg,
            predictor_least_misery=group_predictor_least_misery,
            user_satisfaction=user_satisfaction,
            iterations_to_consider=iterations_to_consider,
        )
        
        self.assertAlmostEqual(
            expected_prediction,
            predictor.get_prediction(group, item),
            places=10
        )