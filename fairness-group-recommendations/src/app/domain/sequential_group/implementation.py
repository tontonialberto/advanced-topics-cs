from typing import List, Tuple
from app.domain.dataset import Dataset, ItemId
from app.domain.group_prediction.group_prediction import Group
from app.domain.recommendations_log import RecommendationsLog
from app.domain.sequential_group.prediction.sequential_group_prediction import SequentialGroupPrediction
from app.domain.sequential_group.sequential_group_recommender import SequentialGroupRecommender


class SequentialGroupRecommenderImpl(SequentialGroupRecommender):
    def __init__(self, dataset: Dataset, predictor: SequentialGroupPrediction) -> None:
        self.__dataset = dataset
        self.__predictor = predictor
        self.__recommendations_log = RecommendationsLog(dataset)
    
    def get_recommendations(self, group: Group, limit: int) -> List[Tuple[ItemId, float]]:
        all_items = self.__recommendations_log.get_unrecommended_items(group)
        item_ratings = [
            (item, self.__predictor.get_prediction(group, item))
            for item in all_items
        ]
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = item_ratings[:limit]
        
        self.__recommendations_log.add_recommendation(group, set(item for item, _ in top_recommendations))
        
        return top_recommendations
    
    def get_previous_recommendations(self, group: Group) -> List[List[ItemId]]:
        return [
            list(recommendation) 
            for recommendation in self.__recommendations_log.get_previous_recommendations(group)
        ]
    