from typing import List
from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset, ItemId
from app.domain.sequential_group.implementation import SequentialGroupRecommenderImpl
from app.domain.sequential_group.prediction.sequential_group_prediction import SequentialGroupPrediction


class TestSequentialGroupRecommenderImpl(TestCase):
    def test_get_recommendations_single_iteration(self):
        group = [1, 2, 3]
        all_items = [1, 2, 3, 4, 5]
        n_top_recommendations = 2
        item_predictions = [5, 4.5, 1, 4.8, 4.9]
        expected_recommendations = [(1, 5), (5, 4.9)]
        
        dataset = Mock(spec=Dataset)
        dataset.get_all_items.return_value = all_items
        predictor = Mock(spec=SequentialGroupPrediction)
        predictor.get_prediction.side_effect = item_predictions
        recommender = SequentialGroupRecommenderImpl(dataset, predictor)
        
        self.assertEqual(
            expected_recommendations,
            recommender.get_recommendations(group, limit=n_top_recommendations)
        )
        
    def test_get_recommendations_should_not_return_previous_recommendations(self):
        group = [1, 2, 3]
        all_items = [1, 2, 3, 4, 5]
        n_top_recommendations = 2
        item_predictions = [5, 4.5, 1, 4.8, 4.9]
        
        dataset = Mock(spec=Dataset)
        dataset.get_all_items.return_value = all_items
        predictor = Mock(spec=SequentialGroupPrediction)
        predictor.get_prediction.side_effect = item_predictions
        recommender = SequentialGroupRecommenderImpl(dataset, predictor)
        
        recommender.get_recommendations(group, limit=n_top_recommendations)
        
        remaining_item_predictions = [4.5, 1, 4.8]
        expected_recommendations = [(4, 4.8), (2, 4.5)]
        dataset.get_all_items.return_value = all_items
        predictor.get_prediction.side_effect = remaining_item_predictions
        
        self.assertEqual(
            expected_recommendations,
            recommender.get_recommendations(group, limit=n_top_recommendations)
        )
        
        dataset.get_all_items.return_value = all_items
        remaining_item_predictions = [1]
        expected_recommendations = [(3, 1)]
        predictor.get_prediction.side_effect = remaining_item_predictions
        
        self.assertEqual(
            expected_recommendations,
            recommender.get_recommendations(group, limit=n_top_recommendations)
        )
        
    def test_get_previous_recommendations(self) -> None:
        group = [1, 2, 3]
        all_items = [1, 2, 3, 4, 5]
        n_top_recommendations = 2
        item_predictions = [5, 4.5, 1, 4.8, 4.9]
        expected_previous_recommendations: List[List[ItemId]] = [[1, 5], [2, 4], [3]]
        
        dataset = Mock(spec=Dataset)
        dataset.get_all_items.return_value = all_items
        predictor = Mock(spec=SequentialGroupPrediction)
        predictor.get_prediction.side_effect = item_predictions
        recommender = SequentialGroupRecommenderImpl(dataset, predictor)
        
        recommender.get_recommendations(group, limit=n_top_recommendations)
        
        self.assertEqual(
            [expected_previous_recommendations[0]],
            recommender.get_previous_recommendations(group)
        )
        
        dataset.get_all_items.return_value = all_items
        remaining_item_predictions = [4.5, 1, 4.8]
        predictor.get_prediction.side_effect = remaining_item_predictions
        recommender.get_recommendations(group, limit=n_top_recommendations)
        
        self.assertEqual(
            [expected_previous_recommendations[0], expected_previous_recommendations[1]],
            recommender.get_previous_recommendations(group)
        )
        
        dataset.get_all_items.return_value = all_items
        remaining_item_predictions = [4.5, 1, 4.8]
        predictor.get_prediction.side_effect = remaining_item_predictions
        recommender.get_recommendations(group, limit=n_top_recommendations)
        
        self.assertEqual(
            expected_previous_recommendations,
            recommender.get_previous_recommendations(group)
        )