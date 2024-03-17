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
        self.__precompute()
        
    def __len__(self) -> int:
        return len(self.__data)
        
    def data(self) -> List[DataRow]:
        return self.__data
    
    def get_average_rating_by_user(self, user: UserId) -> float:
        return self.__average_user_ratings[user]
    
    def get_items_rated_by_both(self, user_a: UserId, user_b: UserId) -> Dict[ItemId, Tuple[float, float]]:
        items_rated_by_a = self.__user_ratings[user_a]
        items_rated_by_b = self.__user_ratings[user_b]
        
        common_items = items_rated_by_a.keys() & items_rated_by_b.keys()
        
        result = {}
        for item in common_items:
            result.update({
                item: (items_rated_by_a[item], items_rated_by_b[item]),
            })
        return result
    
    def get_items_rated_by_any(self, user_a: UserId, user_b: UserId) -> Dict[ItemId, Tuple[float, float]]:
        items_rated_by_a = self.__user_ratings[user_a]
        items_rated_by_b = self.__user_ratings[user_b]
        
        all_items = items_rated_by_a.keys() | items_rated_by_b.keys()
        
        result = {}
        for item in all_items:
            result.update({
                item: (items_rated_by_a.get(item, 0), items_rated_by_b.get(item, 0)),
            })
        return result
    
    def get_ratings_by_user(self, user: UserId) -> List[Tuple[ItemId, float]]:
        return [(item, rating) for item, rating in self.__user_ratings[user].items()]
    
    def get_users_who_rated(self, item: ItemId) -> List[Tuple[UserId, float]]:
        return [
            (user_id, rating)
            for (user_id, item_id, rating) in self.__data
            if item == item_id
        ]
        
    def get_items_not_rated_by_user(self, user: UserId) -> List[ItemId]:
        rated_items = set([item for (item, _) in self.get_ratings_by_user(user)])
        all_items = set(self.get_all_items())
        unrated_items = all_items - rated_items
        return list(unrated_items)
    
    def get_all_users(self) -> List[UserId]:
        return self.__users
    
    def get_all_items(self) -> List[ItemId]:
        return self.__items
    
    def __compute_average_user_ratings(self) -> Dict[UserId, float]:
        all_users = set([user for (user, _, _) in self.__data])
        result = {}
        for user in all_users:
            user_ratings = self.__user_ratings[user].values()
            avg_rating = sum(user_ratings) / len(user_ratings)
            result.update({user: avg_rating})
        return result
    
    def get_first(self, limit: int) -> List[DataRow]:
        return self.__data[:limit]
    
    def get_rating(self, user: UserId, item: ItemId) -> float:
        return self.__user_ratings[user].get(item, 0)
    
    def __compute_user_ratings(self) -> Dict[UserId, Dict[ItemId, float]]:
        all_users = set([user for (user, _, _) in self.__data])
        result = {}
        for user in all_users:
            user_ratings = {
                item: rating
                for (user_id, item, rating) in self.__data
                if user == user_id
            }
            result.update({user: user_ratings})
        return result
    
    def __compute_all_users(self) -> List[UserId]:
        return list(set([user for (user, _, _) in self.__data]))
    
    def __compute_all_items(self) -> List[ItemId]:
        return list(set([item for (_, item, _) in self.__data]))
    
    def __precompute(self) -> None:
        self.__user_ratings = self.__compute_user_ratings()
        self.__average_user_ratings = self.__compute_average_user_ratings()
        self.__users = self.__compute_all_users()
        self.__items = self.__compute_all_items()