from unittest import TestCase
from unittest.mock import Mock
from app.domain.prediction.mean_centered import ALL_NEIGHBORS, MeanCenteredPrediction
from app.domain.similarity.similarity import Similarity
from app.domain.dataset import Dataset

class TestMeanCenteredPrediction(TestCase):
    def test_get_prediction(self) -> None:
        # test: neighbors with average=0 and similarity=1. Should return the average rating for the user
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = [1]
        predictor = MeanCenteredPrediction(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                (2, 2, 5),
                (2, 3, 5),
            ]),
            similarity=similarity,
            num_neighbors=ALL_NEIGHBORS,
            use_absolute_value=True,
        )
        self.assertEqual(
            5,
            predictor.get_prediction(user=1, item=2)
        )
        
        # test: filter only the first neighbor with similarity=1
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = [1, 0.5]
        predictor = MeanCenteredPrediction(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                (2, 2, 3),
                (2, 3, 4),
                
                (3, 2, 5),
            ]),
            similarity=similarity,
            num_neighbors=1,
            use_absolute_value=True,
        )
        self.assertEqual(
            5 + (3 - 4), # average of user1 + (neighbor rating for the item - average of neighbor)
            predictor.get_prediction(user=1, item=2)
        )
        
    def test_get_prediction_considering_most_similar_neighbors(self) -> None:
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = [0.8, 1] # user 3 has the highest similarity
        num_neighbors = 1
        predictor = MeanCenteredPrediction(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                
                (3, 1, 1),
                (3, 2, 5),
            ]),
            similarity=similarity,
            num_neighbors=num_neighbors,
            use_absolute_value=True,
        )
        self.assertEqual(
            5, # only the average of user 1 is considered, since user 3 has not rated item 3
            predictor.get_prediction(user=1, item=3)
        )
        
        similarity.get_similarity.side_effect = [0.8, 1] # user 3 has the highest similarity
        predictor = MeanCenteredPrediction(
            dataset=Dataset([
                (1, 1, 5),
                
                (2, 1, 5),
                
                (3, 1, 1),
                (3, 2, 5),
            ]),
            similarity=similarity,
            num_neighbors=num_neighbors,
            use_absolute_value=True,
        )
        self.assertEqual(
            5 + (5 - 3), # average of user 1 + (rating of user 3 for item 2 - average of user 3)
            predictor.get_prediction(user=1, item=2)
        )
        
        
        