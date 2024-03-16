from typing import Dict, Tuple
from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset, UserId
from app.domain.recommender import Stats
from app.domain.similarity.pearson import PearsonCorrelation
from app.domain.similarity.similarity import Similarity


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
        
    def test_get_user_similarity_matrix(self) -> None:
        dataset = Mock(spec=Dataset)
        dataset.get_all_users.return_value= [1, 2, 3]
        def get_similarity(user_a: UserId, user_b: UserId) -> float:
            mock_similarities = {
                (1, 1): 1,
                (1, 2): 0.5,
                (1, 3): 0,
                
                (2, 1): 0.5,
                (2, 2): 1,
                (2, 3): -1,
                
                (3, 1): 0,
                (3, 2): -1,
                (3, 3): 1,
            }
            return mock_similarities[(user_a, user_b)] # type: ignore
        similarity = Mock(spec=Similarity)
        similarity.get_similarity.side_effect = get_similarity
        stats = Stats(dataset, similarity)
        
        user_similarities: Dict[Tuple[UserId, UserId], float] = stats.get_user_similarity_matrix()
        
        self.assertEqual(9, len(user_similarities))
        self.assertEqual(1, user_similarities[(1, 1)])
        self.assertEqual(0.5, user_similarities[(1, 2)])
        self.assertEqual(0, user_similarities[(1, 3)])
        
        self.assertEqual(0.5, user_similarities[(2, 1)])
        