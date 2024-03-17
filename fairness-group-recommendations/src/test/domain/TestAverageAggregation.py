from unittest import TestCase
from unittest.mock import Mock

from app.domain.average_aggregation import AverageAggregation
from app.domain.dataset import Dataset
from app.domain.prediction.prediction import Prediction


class TestAverageAggregation(TestCase):
    def test_get_prediction(self) -> None:
        """
        Scenario:
        - User 1 has not rated the item.
        - User 2 has rated the item with a score of r_i2.
        - Prediction for User 1 on the item is r_i1.
        - Prediction for the group is the average of r_i1 and r_i2.
        """
        item = 1
        item_ratings = [0, 4]
        group = [1, 2]
        dataset = Mock(spec=Dataset)
        dataset.get_rating.side_effect = item_ratings
        user_predictor = Mock(spec=Prediction)
        user_predictor.get_prediction.return_value = 3 # prediction for user 1 on item
        group_predictor = AverageAggregation(dataset, user_predictor)
        
        prediction = group_predictor.get_prediction(group, item)
        
        self.assertEqual(3.5, prediction)
        