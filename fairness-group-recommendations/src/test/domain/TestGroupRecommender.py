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
        """
        dataset = Mock(spec=Dataset)
        all_items = [1, 2, 3]
        dataset.get_all_items.return_value = all_items
        group = [1, 2]
        group_predictor = Mock(spec=GroupPrediction)
        group_predictor.get_prediction.side_effect = [4, 5, 1] # item 2 is the most preferred, then item 1, then item 3
        recommender = GroupRecommender(dataset, group_predictor)
        
        recommendations = recommender.get_recommendations(group, limit=len(all_items))
        self.assertEqual(
            [(2, 5), (1, 4), (3, 1)],
            recommendations
        )