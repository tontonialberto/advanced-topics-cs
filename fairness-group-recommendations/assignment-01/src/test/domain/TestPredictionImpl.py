from unittest import TestCase
from unittest.mock import Mock
from app.domain.recommender import ALL_NEIGHBORS, PredictionImpl
from app.domain.similarity.similarity import Similarity
from app.domain.dataset import Dataset

class TestPredictionImpl(TestCase):
    def test_get_prediction(self) -> None:
        # test: neighbors with average=0 and similarity=1. Should return the average rating for the user
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = [1]
        predictor = PredictionImpl(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                (2, 2, 5),
                (2, 3, 5),
            ]),
            similarity=similarity,
            num_neighbors=ALL_NEIGHBORS
        )
        self.assertEqual(
            5,
            predictor.get_prediction(user=1, item=2)
        )
        
        # test: filter only the first neighbor with similarity=1
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = [1, 0.5]
        predictor = PredictionImpl(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                (2, 2, 3),
                (2, 3, 4),
                
                (3, 2, 5),
            ]),
            similarity=similarity,
            num_neighbors=1
        )
        self.assertEqual(
            5 + (3 - 4), # average of user1 + (neighbor rating for the item - average of neighbor)
            predictor.get_prediction(user=1, item=2)
        )