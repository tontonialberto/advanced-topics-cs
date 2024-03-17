from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.recommender import PerformanceEvaluator, Prediction, PredictionComparison


class TestPerformanceEvaluator(TestCase):
    def test_get_comparison_by_user_2(self) -> None:
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
        
        evaluations = evaluator.get_comparison_by_user(user=1)
        
        self.assertEqual(2, len(evaluations))
        
        evaluation_a = evaluations["A"]
        predictions_a = evaluation_a.predictions
        
        self.assertEqual(4, len(predictions_a))
        self.assertEqual(0.75, evaluation_a.mean_absolute_error)
        
        prediction_a_item_1 = predictions_a[1]
        self.assertEqual(1, prediction_a_item_1.absolute_error)
        
    def test_get_all_predictions(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 4),
            (1, 3, 3),
            
            (2, 1, 1),
            (2, 2, 2),
        ])
        predictor = Mock(spec=Prediction)
        predictor.get_prediction.side_effect = [
            # Second movie is better suited for user 1
            1,
            2,
            5,
            
            # First movie is better suited for user 2
            5,
            1,
            4,
        ]
        
        evaluator = PerformanceEvaluator([("A", predictor)], dataset)
        
        self.assertEqual(
            {
                1: [(1, 1), (2, 2), (3, 5)],
                2: [(1, 5), (2, 1), (3, 4)],
            },
            evaluator.get_all_predictions(predictor)
        )