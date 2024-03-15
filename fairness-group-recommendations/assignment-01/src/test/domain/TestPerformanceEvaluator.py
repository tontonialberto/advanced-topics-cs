from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.recommender import PerformanceEvaluator, Prediction, PredictionComparison


class TestPerformanceEvaluator(TestCase):
    def test_get_comparison_by_user(self) -> None:
        user = 1
        dataset = Dataset(data=[
            (user, 1, 5),
            (user, 2, 4),
            (user, 3, 3),
            (user, 4, 5),
        ])
        predictor_a = Mock(spec=Prediction)
        predictor_a.get_prediction.side_effect = [4, 3, 2, 5]
        predictor_b = Mock(spec=Prediction)
        predictor_b.get_prediction.side_effect = [1, 1, 4, 1]
        evaluator = PerformanceEvaluator([("A", predictor_a), ("B", predictor_b)], dataset)
        
        comparison = evaluator.get_comparison_by_user(1)
        
        self.assertEqual(len(comparison), 4)
        
        self.assertEqual([1, 4], comparison[1].errors)
        self.assertEqual([1, 3], comparison[2].errors)
        self.assertEqual([1, 1], comparison[3].errors)
        self.assertEqual([0, 4], comparison[4].errors)