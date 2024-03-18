from app.domain.dataset import ItemId
from app.domain.disagreement import Disagreement
from app.domain.group_prediction import Group, GroupPrediction


class Consensus(GroupPrediction):
    def __init__(self, group_predictor: GroupPrediction, disagreement: Disagreement, weight_prediction: float, weight_disagreement: float) -> None:
        self.__group_predictor = group_predictor
        self.__disagreement = disagreement
        self.__weight_prediction = weight_prediction
        self.__weight_disagreement = weight_disagreement
    
    def get_prediction(self, group: Group, item: ItemId) -> float:
        prediction = self.__group_predictor.get_prediction(group, item)
        disagreement = self.__disagreement.get_disagreement(group, item)
        return (self.__weight_prediction * prediction) + (self.__weight_disagreement * (1 - disagreement))