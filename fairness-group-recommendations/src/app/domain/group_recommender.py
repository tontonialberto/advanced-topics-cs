from typing import Dict, List, Set, Tuple
from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction.group_prediction import Group, GroupPrediction

class GroupRecommender:
    def __init__(self, dataset: Dataset, predictor: GroupPrediction, exclude_previous: bool) -> None:
        """
        Initializes a GroupRecommender object.

        Parameters:
        - dataset: The dataset containing user-item interactions.
        - predictor: The predictor used for making group recommendations.
        - exclude_previous: Flag indicating whether to exclude previous recommendations.
        """
        
        self.__dataset = dataset
        self.__group_predictor = predictor
        self.__exclude_previous = exclude_previous
        self.__previous_recommendations: Dict[str, Set[ItemId]] = {} # ugly hack: dict keys are the string representations of the group as a Set. Reason: set is unhashable.
    
    def get_recommendations(self, group: Group, limit: int) -> List[Tuple[ItemId, float]]:
        all_items = self.__filter_previously_recommended_items(group)
        
        recommendations = [
            (item, self.__group_predictor.get_prediction(group, item))
            for item in all_items
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:limit]
        
        self.__update_previous_recommendations(group, top_recommendations)
        
        return top_recommendations
    
    def __filter_previously_recommended_items(self, group: Group) -> Set[ItemId]:
        all_items = set(self.__dataset.get_all_items())
        if self.__exclude_previous:
            group_key = str(set(group))
            previous_recommendations = self.__previous_recommendations.get(group_key, set())
            all_items -= set(previous_recommendations)
        return all_items
    
    def __update_previous_recommendations(self, group: Group, recommendations: List[Tuple[ItemId, float]]) -> None:
        if self.__exclude_previous:
            group_key = str(set(group))
            if self.__previous_recommendations.get(group_key):
                self.__previous_recommendations[group_key].update(item for item, _ in recommendations)
            else:
                self.__previous_recommendations[group_key] = set(item for item, _ in recommendations)
    
    