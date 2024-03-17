from unittest import TestCase

from app.domain.dataset import Dataset


class TestDataset(TestCase):
    def test_get_average_rating_by_user(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 4),
            (1, 3, 3),
            
            (2, 1, 1),
            (2, 2, 2),
        ])
        self.assertEqual(4, dataset.get_average_rating_by_user(1))
        self.assertAlmostEqual(1.5, dataset.get_average_rating_by_user(2), 1)
        
    def test_get_items_rated_by_both(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 4),
            
            (2, 1, 1),
            (2, 3, 1),
        ])
        expected_items = {
            1: (5, 1)
        }
        self.assertEqual(expected_items, dataset.get_items_rated_by_both(1, 2))
        
    def test_get_items_rated_by_any(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 4),
            (1, 4, 1),
            
            (2, 1, 1),
            (2, 3, 3),
            (2, 4, 1),
        ])
        
        self.assertEqual(
            {
                1: (5, 1),
                2: (4, 0),
                3: (0, 3),
                4: (1, 1),
            },
            dataset.get_items_rated_by_any(1, 2)
        )
        
    def test_get_users_who_rated(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(
            [],
            dataset.get_users_who_rated(item=0)
        )
        
        self.assertEqual(
            [(1, 5)],
            dataset.get_users_who_rated(item=2)
        )
        
        self.assertEqual(
            [(1, 5), (2, 1)],
            dataset.get_users_who_rated(item=1)
        )
        
    def test_get_items_not_rated_by_user(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(
            [2],
            dataset.get_items_not_rated_by_user(2)
        )
        
    def test_get_all_users(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(
            [1, 2],
            dataset.get_all_users()
        )
        
    def test_get_all_items(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(
            [1, 2],
            dataset.get_all_items()
        )
    
    def test_len(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(3, len(dataset))
        
    def test_get_first(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            (1, 2, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(
            [(1, 1, 5), (1, 2, 5)],
            dataset.get_first(2)
        )
        
    def test_get_rating(self) -> None:
        dataset = Dataset(data=[
            (1, 1, 5),
            
            (2, 1, 1),
        ])
        
        self.assertEqual(5, dataset.get_rating(1, 1))
        self.assertEqual(1, dataset.get_rating(2, 1))
        self.assertEqual(0, dataset.get_rating(1, 3))