from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.prediction.prediction import Prediction
from app.domain.recommender import Recommender


class TestRecommender(TestCase):
    def test_get_recommendations(self) -> None:
        dataset = Mock(spec=Dataset)
        dataset.get_items_not_rated_by_user.return_value = [1, 2]
        predictor = Mock(spec=Prediction)
        predictor.get_prediction.side_effect = [
            # Second movie is better suited for user
            1,
            5
        ]
        recommender = Recommender(dataset, predictor)
        
        self.assertEqual(
            [(2, 5), (1, 1)],
            recommender.get_recommendations(user=0, limit=2)
        )