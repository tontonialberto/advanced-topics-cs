from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.group_prediction.group_prediction import GroupPrediction
from app.domain.group_recommender import GroupRecommender


class TestGroupRecommender(TestCase):
    def test_get_recommendations(self) -> None:
        """
        Scenario:
        - Three items in the dataset.
        - Two users in the group.
        - The recommender presents the items according to the group's preferences.
        - Two iterations (ie. two calls to the group recommender).
        """
        all_items = [1, 2, 3]
        group_predictions = [4, 5, 1] # item 2 is the most preferred, then item 1, then item 3
        
        dataset = Mock(spec=Dataset)
        dataset.get_all_items.return_value = all_items
        group_predictor = Mock(spec=GroupPrediction)
        group_predictor.get_prediction.side_effect = group_predictions
        recommender = GroupRecommender(dataset, group_predictor, exclude_previous=False)
        
        expected_recommendations = [(2, 5), (1, 4), (3, 1)]
        n_recommendations = 3
        group = [1, 2]
        
        recommendations = recommender.get_recommendations(group, limit=n_recommendations)
        self.assertEqual(
            expected_recommendations,
            recommendations
        )
        
        dataset.get_all_items.return_value = all_items
        group_predictor.get_prediction.side_effect = group_predictions
        # Recommendations do not change unless specified
        recommendations = recommender.get_recommendations(group, limit=n_recommendations)
        self.assertEqual(
            expected_recommendations,
            recommendations,
        )
        
    def test_get_recommendations_excluding_previous_ones(self) -> None:
        group = [1, 2]
        n_recommendations = 3
        all_items = [1, 2, 3, 4, 5]
        group_predictions = [5, 5, 4, 1, 1] # Item 4 and 5 should be excluded from the first "iteration".
        
        dataset = Mock(spec=Dataset)
        dataset.get_all_items.return_value = all_items
        group_predictor = Mock(spec=GroupPrediction)
        group_predictor.get_prediction.side_effect = group_predictions
        recommender = GroupRecommender(dataset, group_predictor, exclude_previous=True)
        
        recommender.get_recommendations(group, limit=n_recommendations)
        
        dataset.get_all_items.return_value = all_items
        group_predictor.get_prediction.side_effect = [1, 1] # As before, Item 4 and 5 will have the same rating.
        recommendations = recommender.get_recommendations(group, limit=n_recommendations)
        
        expected_recommendations = [(4, 1), (5, 1)]
        self.assertEqual(
            expected_recommendations,
            recommendations,
        )