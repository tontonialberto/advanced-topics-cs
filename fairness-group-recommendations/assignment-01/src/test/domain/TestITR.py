from unittest import TestCase
from unittest.mock import Mock

from app.domain.dataset import Dataset
from app.domain.similarity.itr import ITR


class TestITR(TestCase):
    """
    Tests are based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9129689"
    """
    
    def setUp(self) -> None:
        self.dataset = Dataset(data=[
            (1, 1, 4),
            (1, 2, 3),
            (1, 3, 5),
            (1, 4, 4),
            (1, 5, 2),
            
            (2, 1, 5),
            (2, 2, 1),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 4),
        ])

    def test_get_similarity(self) -> None:
        itr = ITR(dataset=Dataset(data=[
            (1, 1, 5),
            
            (2, 1, 5),
        ]))
        self.assertAlmostEqual(0.5, itr.get_similarity(1, 2), 2)
        
        itr = ITR(dataset=Dataset(data=[
            (1, 1, 5),
        ]))
        self.assertAlmostEqual(0.5, itr.get_similarity(1, 1), 2)
        
    def test_get_similarity_triangle_improved(self) -> None:
        itr = ITR(dataset=self.dataset)
        union_items_1_2 = self.dataset.get_items_rated_by_any(user_a=1, user_b=2)
        self.assertAlmostEqual(
            0.523748,
            itr.get_similarity_triangle_improved(union_items_1_2),
            5
        )
        
    def test_get_similarity_urp(self) -> None:
        itr = ITR(dataset=self.dataset)
        union_items_1_2 = self.dataset.get_items_rated_by_any(user_a=1, user_b=2)
        self.assertAlmostEqual(
            0.024244,
            itr.get_similarity_urp(union_items=union_items_1_2),
            5
        )
        
    def test_name(self) -> None:
        itr = ITR(dataset=Mock())
        self.assertEqual(ITR.__name__, itr.name)