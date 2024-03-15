from unittest import TestCase

from app.domain.dataset import Dataset
from app.domain.recommender import Stats
from app.domain.similarity.pearson import PearsonCorrelation


class TestStats(TestCase):
    def test_get_most_similar_users(self) -> None:
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
        stats = Stats(dataset, PearsonCorrelation(dataset))
        
        most_similar_users = stats.get_most_similar_users(user=0, limit=4)
        
        self.assertEqual(4, len(most_similar_users))
        
        most_similar_user, highest_similarity = most_similar_users[0]
        self.assertEqual(1, most_similar_user)
        self.assertAlmostEqual(
            0.839,
            highest_similarity,
            3
        )