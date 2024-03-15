from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

UserId = int
ItemId = int

DataRow = Tuple[UserId, ItemId, float]

@dataclass
class Dataset:
    def __init__(self, data: List[DataRow]) -> None:
        self.__data = data
        self.__average_user_ratings = self.__compute_average_user_ratings()
        
    def data(self) -> List[DataRow]:
        return self.__data
    
    def get_average_rating_by_user(self, user: UserId) -> float:
        return self.__average_user_ratings[user]
    
    def get_items_rated_by_both(self, user_a: UserId, user_b: UserId) -> Dict[ItemId, Tuple[float, float]]:
        items_rated_by_a: Dict[int, float] = {}
        items_rated_by_b: Dict[int, float] = {}
        for user, item, rating in self.__data:
            if user == user_a:
                items_rated_by_a.update({item: rating})
            elif user == user_b:
                items_rated_by_b.update({item: rating})
        
        common_items = items_rated_by_a.keys() & items_rated_by_b.keys()
        
        result = {}
        for item in common_items:
            result.update({
                item: (items_rated_by_a[item], items_rated_by_b[item]),
            })
        return result
    
    def get_items_rated_by_any(self, user_a: UserId, user_b: UserId) -> Dict[ItemId, Tuple[float, float]]:
        items_rated_by_a: Dict[int, float] = {}
        items_rated_by_b: Dict[int, float] = {}
        for user, item, rating in self.__data:
            if user == user_a:
                items_rated_by_a.update({item: rating})
            elif user == user_b:
                items_rated_by_b.update({item: rating})
        
        all_items = items_rated_by_a.keys() | items_rated_by_b.keys()
        
        result = {}
        for item in all_items:
            result.update({
                item: (items_rated_by_a.get(item, 0), items_rated_by_b.get(item, 0)),
            })
        return result
    
    def get_ratings_by_user(self, user: UserId) -> List[Tuple[ItemId, float]]:
        ratings = [(item, rating) for (user_id, item, rating) in self.__data if user == user_id]
        return ratings
    
    def get_users_who_rated(self, item: ItemId) -> List[Tuple[UserId, float]]:
        return [
            (user_id, rating)
            for (user_id, item_id, rating) in self.__data
            if item == item_id
        ]
        
    def get_items_not_rated_by_user(self, user: UserId) -> List[ItemId]:
        rated_items = set([item for (item, _) in self.get_ratings_by_user(user)])
        all_items = set([item for (_, item, _) in self.__data])
        unrated_items = all_items - rated_items
        return list(unrated_items)
    
    def __compute_average_user_ratings(self) -> Dict[UserId, float]:
        all_users = set([user for (user, _, _) in self.__data])
        result = {}
        for user in all_users:
            user_ratings = [
                rating 
                for (user_id, _, rating) in self.__data
                if user == user_id
            ]
            avg_rating = sum(user_ratings) / len(user_ratings)
            result.update({user: avg_rating})
        return result