from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.similarity.jaccard import Jaccard


class TestJaccardSimilarity(TestCase):
    def test_get_similarity(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 1),
            (1, 2, 4),
            
            (2, 1, 5),
            (2, 2, 4),
            
            (3, 1, 5),
            
            (4, 3, 5),
        ])
        
        jaccard = Jaccard(dataset)
        self.assertEqual(1, jaccard.get_similarity(user_a=1, user_b=2))
        self.assertEqual(0.5, jaccard.get_similarity(user_a=1, user_b=3))
        self.assertEqual(0, jaccard.get_similarity(user_a=1, user_b=4))
    
    def test_name(self) -> None:
        itr = Jaccard(dataset=Mock())
        self.assertEqual(Jaccard.__name__, itr.name)