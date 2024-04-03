from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Tuple
from app.domain.group_prediction.average_aggregation import AverageAggregation
from app.domain.group_prediction.disagreement.average_pairwise_disagreement import AveragePairwiseDisagreement
from app.domain.group_prediction.consensus import Consensus
from app.domain.group_recommender import GroupRecommender
from app.domain.group_prediction.least_misery_aggregation import LeastMiseryAggregation
from app.domain.prediction.mean_centered import ALL_NEIGHBORS, MeanCenteredPrediction
from app.domain.prediction.prediction import Prediction
from app.domain.sequential_group.implementation import SequentialGroupRecommenderImpl
from app.domain.sequential_group.prediction.multi_iter_sequential_hybrid_aggregation import MultiIterSequentialHybridAggregation
from app.domain.sequential_group.sequential_group_recommender import SequentialGroupRecommender
from app.domain.similarity.stats import Stats
from app.domain.user_satisfaction import UserSatisfaction
from app.result_saver.csv_result_saver import CsvResultSaver
from app.ui.cli import start_cli_menu
from app.data_loader.file_data_loader import FileDataLoader
from app.domain.dataset import Dataset
from app.domain.recommender import PerformanceEvaluator, Recommender
from app.domain.similarity.cached import CachedSimilarity
from app.domain.similarity.itr import ITR
from app.domain.similarity.jaccard import Jaccard
from app.domain.similarity.pearson import PearsonCorrelation
from app.domain.similarity.similarity import Similarity


@dataclass
class SystemOptions:
    similarity_func: str
    prediction_func: str
    num_neighbors: int
    consensus_weight_disagreement: float
    seq_most_recent_iterations: int

def parse_system_options() -> SystemOptions:
    SIMILARITY_FUNC = os.environ.get("SIMILARITY_FUNC")
    if SIMILARITY_FUNC not in ["itr", "pearson", "jaccard"]:
        SIMILARITY_FUNC = "pearson"
    
    PREDICTION_FUNC = os.environ.get("PREDICTION_FUNC")
    if PREDICTION_FUNC not in ["mean_centered_abs", "mean_centered_no_abs"]:
        PREDICTION_FUNC = "mean_centered_abs"
    
    try:
        NUM_NEIGHBORS = int(os.environ.get("NUM_NEIGHBORS", ALL_NEIGHBORS))
    except ValueError:
        NUM_NEIGHBORS = ALL_NEIGHBORS
        
    try:
        consensus_weight_disagreement_int = int(os.environ.get("CONSENSUS_WEIGHT_DISAGREEMENT", 2))
        if consensus_weight_disagreement_int < 0 or consensus_weight_disagreement_int > 10:
            raise ValueError()
        
        CONSENSUS_WEIGHT_DISAGREEMENT = int(os.environ.get("CONSENSUS_WEIGHT_DISAGREEMENT", 2)) / 10
    except ValueError:
        CONSENSUS_WEIGHT_DISAGREEMENT = 0.2
    
    try:
        SEQ_MOST_RECENT_ITERATIONS = int(os.environ.get("SEQ_MOST_RECENT_ITERATIONS", 2))
    except ValueError:
        SEQ_MOST_RECENT_ITERATIONS = 2
        
    system_options = SystemOptions(
        similarity_func=SIMILARITY_FUNC,
        prediction_func=PREDICTION_FUNC,
        num_neighbors=NUM_NEIGHBORS,
        consensus_weight_disagreement=CONSENSUS_WEIGHT_DISAGREEMENT,
        seq_most_recent_iterations=SEQ_MOST_RECENT_ITERATIONS,
    )
    return system_options

def create_similarity_functions(dataset: Dataset) -> Dict[str, Similarity]:
    return {
        "pearson": CachedSimilarity(PearsonCorrelation(dataset)),
        "itr": CachedSimilarity(ITR(dataset)),
        "jaccard": CachedSimilarity(Jaccard(dataset)),
    }    
    
def main() -> None:
    DATASET_FILE_PATH = Path.cwd().parent / "resources" / "ml-latest-small" / "ratings.csv"
    RESULTS_PATH = Path.cwd().parent / "results"
    
    system_options = parse_system_options()
    
    print(f"Using similarity function: {system_options.similarity_func}.")
    print(f"Using prediction function: {system_options.prediction_func}.")
    print(f"Considering {f'all' if system_options.num_neighbors == ALL_NEIGHBORS else f'only {system_options.num_neighbors} most similar'} neighbors for computing predictions.")
    print(f"Using consensus with weight for disagreement: {system_options.consensus_weight_disagreement}.")
    print(f"Considering {system_options.seq_most_recent_iterations} most recent iterations for sequential group recommender.")
    
    print("Loading dataset...")
    
    loader = FileDataLoader(DATASET_FILE_PATH)
    dataset = loader.load()
    
    print(f"Loaded '{DATASET_FILE_PATH.absolute().as_posix()}'.")
    
    similarity_functions = create_similarity_functions(dataset)
    chosen_similarity = similarity_functions[system_options.similarity_func]

    if system_options.prediction_func == "mean_centered_abs":
        USE_MEAN_CENTERED_PRED_ABSOLUTE_VALUE = True
    else:
        USE_MEAN_CENTERED_PRED_ABSOLUTE_VALUE = False
        
    predictor = MeanCenteredPrediction(dataset, chosen_similarity, system_options.num_neighbors, use_absolute_value=USE_MEAN_CENTERED_PRED_ABSOLUTE_VALUE)
    predictors_for_comparison: List[Tuple[str, Prediction]] = [
        (name, MeanCenteredPrediction(dataset, similarity, system_options.num_neighbors, use_absolute_value=USE_MEAN_CENTERED_PRED_ABSOLUTE_VALUE))
        for name, similarity in similarity_functions.items()
    ]
    
    stats = Stats(dataset, chosen_similarity)
    recommender = Recommender(dataset, predictor)
    
    evaluator = PerformanceEvaluator(predictors_for_comparison, dataset)
        
    def file_writer(output_path: Path, content: str) -> None:
        with open(output_path, "w") as file:
            file.write(content)
    
    result_saver = CsvResultSaver(file_writer)
    
    group_predictor_avg = AverageAggregation(dataset, predictor)
    recommender_avg = GroupRecommender(dataset, group_predictor_avg, exclude_previous=False)
    
    group_predictor_least_misery = LeastMiseryAggregation(dataset, predictor)
    recommender_least_misery = GroupRecommender(dataset, group_predictor_least_misery, exclude_previous=False)
    
    disagreement = AveragePairwiseDisagreement(dataset, predictor)
    group_predictor_consensus = Consensus(
        group_predictor_avg,
        disagreement,
        weight_disagreement=system_options.consensus_weight_disagreement,
    )
    recommender_consensus = GroupRecommender(dataset, group_predictor_consensus, exclude_previous=False)
    
    realistic_group_recommender_avg = GroupRecommender(dataset, group_predictor_avg, exclude_previous=True)
    realistic_group_recommender_least_misery = GroupRecommender(dataset, group_predictor_least_misery, exclude_previous=True)
    
    sequential_group_recommender: SequentialGroupRecommender
    def get_previous_recommendations(group: List[int]) -> List[List[int]]:
        return sequential_group_recommender.get_previous_recommendations(group)
    
    user_satisfaction = UserSatisfaction(recommender, predictor, dataset)
    sequential_group_predictor = MultiIterSequentialHybridAggregation(
        get_previous_recommendations,
        group_predictor_avg,
        group_predictor_least_misery,
        user_satisfaction,
        iterations_to_consider=system_options.seq_most_recent_iterations,
    )
    sequential_group_recommender = SequentialGroupRecommenderImpl(
        dataset,
        sequential_group_predictor
    )
    
    start_cli_menu(
        dataset, 
        stats, 
        recommender, 
        evaluator, 
        predictor, 
        chosen_similarity, 
        result_saver, 
        RESULTS_PATH,
        recommender_avg,
        recommender_least_misery,
        recommender_consensus,
        disagreement,
        realistic_group_recommender_avg,
        realistic_group_recommender_least_misery,
        sequential_group_recommender,
        user_satisfaction,
    )

if __name__ == "__main__":
    main()