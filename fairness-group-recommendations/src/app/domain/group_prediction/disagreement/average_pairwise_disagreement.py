import itertools
from typing import List
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.group_prediction.disagreement.disagreement import Disagreement
from app.domain.group_prediction.group_prediction import Group
from app.domain.prediction.prediction import Prediction


class AveragePairwiseDisagreement(Disagreement):
    def __init__(self, dataset: Dataset, predictor: Prediction) -> None:
        self.__dataset = dataset
        self.__predictor = predictor
        
    def __get_individual_relevance(self, user: UserId, item: ItemId) -> float:
        rating = self.__dataset.get_rating(user, item)
        if rating == 0:
            rating = self.__predictor.get_prediction(user, item)
        return rating

    def get_disagreement(self, group: Group, item: ItemId) -> float:
        user_pairs = itertools.combinations(group, 2)
        pair_disagreements: List[float] = []
        for user_a, user_b in user_pairs:
            # rating_a = self.__dataset.get_rating(user_a, item)
            # if rating_a == 0:
            #     rating_a = self.__predictor.get_prediction(user_a, item)
            # rating_b = self.__dataset.get_rating(user_b, item)
            # if rating_b == 0:
            #     rating_b = self.__predictor.get_prediction(user_b, item)
            rating_a = self.__get_individual_relevance(user_a, item)
            rating_b = self.__get_individual_relevance(user_b, item)
            pair_disagreements.append(abs(rating_a - rating_b))
        numerator = 2 * sum(pair_disagreements)
        group_size = len(group)
        denominator = group_size * (group_size - 1)
        return numerator / denominator