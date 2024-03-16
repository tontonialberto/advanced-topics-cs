from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.similarity.pearson import PearsonCorrelation


class TestPearsonCorrelation(TestCase):
    def test_get_similarity(self) -> None:
        dataset = Dataset([
            # ratings for Alice
            (0, 1, 5),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 4),
            
            # ratings for User1
            (1, 1, 3),
            (1, 2, 1),
            (1, 3, 2),
            (1, 4, 3),
            (1, 5, 3), 
            
            # ratings for User2
            (2, 1, 4),
            (2, 2, 3),
            (2, 3, 4),
            (2, 4, 3),
            (2, 5, 5),
            
            
            # ratings for User3
            (3, 1, 3),
            (3, 2, 3),
            (3, 3, 1),
            (3, 4, 5),
            (3, 5, 4),
            
            # ratings for User 4
            (4, 1, 1),
            (4, 2, 5),
            (4, 3, 5),
            (4, 4, 2),
            (4, 5, 1),
        ])
        
        pearson_correlation = PearsonCorrelation(dataset)
        
        # Similarity between a user and itself
        self.assertAlmostEqual(
            1.0,
            pearson_correlation.get_similarity(user_a=0, user_b=0),
            20,
        )
        
        # Similarity between Alice and User1
        self.assertAlmostEqual(
            0.839, 
            pearson_correlation.get_similarity(user_a=0, user_b=1), 
            3
        )
        
        # Similarity between Alice and User2
        self.assertAlmostEqual(
            0.606,
            pearson_correlation.get_similarity(user_a=0, user_b=2),
            3
        )
        
        self.assertAlmostEqual(
            0.0,
            pearson_correlation.get_similarity(user_a=0, user_b=3),
            1
        )
        
        self.assertAlmostEqual(
            -0.768,
            pearson_correlation.get_similarity(user_a=0, user_b=4),
            3
        )
    
    def test_name(self) -> None:
        itr = PearsonCorrelation(dataset=Mock())
        self.assertEqual(PearsonCorrelation.__name__, itr.name)