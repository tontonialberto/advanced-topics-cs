from typing import Dict, List, Set

from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction.group_prediction import Group


class RecommendationsLog:
    """
    Collection used to log recommendations made to groups.
    
    Useful to avoid recommending the same items to the same group multiple times.
    """
    
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset
        self.__recommendations: Dict[str, List[Set[ItemId]]] = {}
        
    def add_recommendation(self, group: Group, recommendation: Set[ItemId]) -> None:
        group_key = str(set(group))
        if self.__recommendations.get(group_key):
            self.__recommendations[group_key].append(recommendation)
        else:
            self.__recommendations[group_key] = [recommendation]
    
    def get_unrecommended_items(self, group: Group) -> Set[ItemId]:
        group_key = str(set(group))
        previous_recommendations = set(
            item
            for recommendation in self.__recommendations.get(group_key, [])
            for item in recommendation
        )
        items = set(self.__dataset.get_all_items()) - previous_recommendations
        return items
    
    def get_previous_recommendations(self, group: Group) -> List[Set[ItemId]]:
        """Note: The relative position of the items in the recommendations is not preserved."""
        group_key = str(set(group))
        previous_recommendations = self.__recommendations.get(group_key, [])
        return previous_recommendations