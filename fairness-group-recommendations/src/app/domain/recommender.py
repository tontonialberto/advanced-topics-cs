from dataclasses import dataclass
from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.prediction.prediction import Prediction

    
class Recommender: 
    def __init__(self, dataset: Dataset, predictor: Prediction) -> None:
        self.__dataset = dataset
        self.__predictor = predictor
    
    def get_recommendations(self, user: UserId, limit: int) -> List[Tuple[ItemId, float]]:
        unrated_items = self.__dataset.get_items_not_rated_by_user(user)
        predicted_ratings = [
            (item, self.__predictor.get_prediction(user, item))
            for item in unrated_items
        ]
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return predicted_ratings[:limit]

@dataclass
class PredictionComparison:
    item: ItemId
    actual_rating: float
    predicted_ratings: List[float]
    errors: List[float]
    
PredictorName = str

@dataclass
class ItemPrediction:
    item: ItemId
    prediction: float
    actual_rating: float
    
    @property
    def absolute_error(self) -> float:
        return abs(self.actual_rating - self.prediction)
    

@dataclass
class Evaluation:
    predictions: Dict[ItemId, ItemPrediction]
    
    @property
    def mean_absolute_error(self) -> float:
        return sum([prediction.absolute_error for prediction in self.predictions.values()]) / len(self.predictions)

class PerformanceEvaluator:
    def __init__(self, predictors: List[Tuple[str, Prediction]], dataset: Dataset) -> None:
        self.__predictors = predictors
        self.__dataset = dataset
    
    def get_comparison_by_user(self, user: UserId) -> Dict[PredictorName, Evaluation]:
        actual_ratings = self.__dataset.get_ratings_by_user(user)
        comparisons = {}
        for predictor_name, predictor in self.__predictors:
            item_predictions = {
                item:
                ItemPrediction(
                    item=item,
                    prediction=predictor.get_prediction(user, item),
                    actual_rating=rating,
                )
                for item, rating in actual_ratings
            }
            evaluation = Evaluation(item_predictions)
            comparisons.update({predictor_name: evaluation})
        return comparisons
    
    def get_all_predictions(self, predictor: Prediction) -> Dict[UserId, List[Tuple[ItemId, float]]]:
        all_users = self.__dataset.get_all_users()
        result = {}
        for user in all_users:
            result[user] = [
                (item, predictor.get_prediction(user, item))
                for item in self.__dataset.get_all_items()
            ]
        return result
    
    @property
    def predictor_names(self) -> List[str]:
        return [name for (name, _) in self.__predictors]