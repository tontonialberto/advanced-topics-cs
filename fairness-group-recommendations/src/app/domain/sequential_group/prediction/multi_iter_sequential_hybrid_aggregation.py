from typing import Callable, Dict, List, Tuple
from app.domain.dataset import ItemId
from app.domain.group_prediction.group_prediction import Group, GroupPrediction
from app.domain.sequential_group.prediction.sequential_group_prediction import SequentialGroupPrediction
from app.domain.user_satisfaction import UserSatisfaction


class MultiIterSequentialHybridAggregation(SequentialGroupPrediction):
    def __init__(
            self, 
            get_previous_recommendations: Callable[[Group], List[List[ItemId]]],
            predictor_average: GroupPrediction, 
            predictor_least_misery: GroupPrediction,
            user_satisfaction: UserSatisfaction,
            iterations_to_consider: int) -> None:
        self.__get_previous_recommendations = get_previous_recommendations
        self.__predictor_average = predictor_average
        self.__predictor_least_misery = predictor_least_misery
        self.__user_satisfaction = user_satisfaction
        self.__iterations_to_consider = iterations_to_consider
        
    def get_prediction(self, group: Group, item: ItemId) -> float:
        average_prediction = self.__predictor_average.get_prediction(group, item)
        least_misery_prediction = self.__predictor_least_misery.get_prediction(group, item)
        
        group_recommendations = self.__get_previous_recommendations(group)
        
        if len(group_recommendations) < self.__iterations_to_consider: # Initial iterations
            disagreement = 0
        else:
            user_satisfactions: List[List[float]] = [] # for each iteration, users satisfactions
            for group_recommendation in group_recommendations[-self.__iterations_to_consider:]:
                iteration_user_satisfations = [
                    self.__user_satisfaction.get_satisfaction(user, group_recommendation)
                    for user in group
                ]
                user_satisfactions.append(iteration_user_satisfations)
            
            # Disagreement will be the maximum disagreement among the last considered iterations
            disagreements = [
                max(satisfactions) - min(satisfactions)
                for satisfactions in user_satisfactions
            ]
            disagreement = max(disagreements)
            
        prediction = (1 - disagreement) * average_prediction + disagreement * least_misery_prediction
    
        return prediction