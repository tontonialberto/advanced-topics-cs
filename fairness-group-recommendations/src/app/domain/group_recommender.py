from typing import List, Tuple
from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction import Group, GroupPrediction

class GroupRecommender:
    def __init__(self, dataset: Dataset, predictor: GroupPrediction) -> None:
        self.__dataset = dataset
        self.__group_predictor = predictor
    
    def get_recommendations(self, group: Group, limit: int) -> List[Tuple[ItemId, float]]:
        all_items = self.__dataset.get_all_items()
        recommendations = [
            (item, self.__group_predictor.get_prediction(group, item))
            for item in all_items
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:limit]