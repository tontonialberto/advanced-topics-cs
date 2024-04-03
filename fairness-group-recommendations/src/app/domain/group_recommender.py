from typing import Dict, List, Set, Tuple
from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction.group_prediction import Group, GroupPrediction
from app.domain.recommendations_log import RecommendationsLog

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
        self.__recommendations_log = RecommendationsLog(dataset)
    
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
            all_items = self.__recommendations_log.get_unrecommended_items(group)
        return all_items
    
    def __update_previous_recommendations(self, group: Group, recommendations: List[Tuple[ItemId, float]]) -> None:
        if self.__exclude_previous:
            recommendation = set(item for item, _ in recommendations)
            self.__recommendations_log.add_recommendation(group, recommendation)
    
    