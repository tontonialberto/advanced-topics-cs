from pathlib import Path
import os
from typing import Dict, List, Tuple
from app.domain.prediction.mean_centered import ALL_NEIGHBORS, MeanCenteredPrediction
from app.domain.prediction.prediction import Prediction
from app.result_saver.csv_result_saver import CsvResultSaver
from app.ui.cli import start_cli_menu
from app.data_loader.file_data_loader import FileDataLoader
from app.domain.dataset import Dataset
from app.domain.recommender import PerformanceEvaluator, Recommender, Stats
from app.domain.similarity.cached import CachedSimilarity
from app.domain.similarity.itr import ITR
from app.domain.similarity.jaccard import Jaccard
from app.domain.similarity.pearson import PearsonCorrelation
from app.domain.similarity.similarity import Similarity


def create_similarity_functions(dataset: Dataset) -> Dict[str, Similarity]:
    return {
        "pearson": CachedSimilarity(PearsonCorrelation(dataset)),
        "itr": CachedSimilarity(ITR(dataset)),
        "jaccard": CachedSimilarity(Jaccard(dataset)),
    }

def main() -> None:
    DATASET_FILE_PATH = Path.cwd().parent / "resources" / "ml-latest-small" / "ratings.csv"
    SIMILARITY_FUNC = os.environ.get("SIMILARITY_FUNC")
    if SIMILARITY_FUNC not in ["itr", "pearson", "jaccard"]:
        SIMILARITY_FUNC = "pearson"
    NUM_NEIGHBORS = int(os.environ.get("NUM_NEIGHBORS", ALL_NEIGHBORS))
    RESULTS_PATH = Path.cwd().parent / "results"
    
    print(f"Using similarity function: {SIMILARITY_FUNC}.")
    print(f"Considering {f'only {NUM_NEIGHBORS} most similar' if ALL_NEIGHBORS != -1 else 'all'} neighbors for computing predictions.")
    print("Loading dataset...")
    loader = FileDataLoader(DATASET_FILE_PATH)
    dataset = loader.load()
    
    similarity_functions = create_similarity_functions(dataset)
    chosen_similarity = similarity_functions[SIMILARITY_FUNC]

    stats = Stats(dataset, chosen_similarity)
    predictor = MeanCenteredPrediction(dataset, chosen_similarity, NUM_NEIGHBORS)
    recommender = Recommender(dataset, predictor)
    
    predictors_for_comparison: List[Tuple[str, Prediction]] = [
        (name, MeanCenteredPrediction(dataset, similarity, NUM_NEIGHBORS))
        for name, similarity in similarity_functions.items()
    ]
    
    evaluator = PerformanceEvaluator(predictors_for_comparison, dataset)
    
    print(f"Loaded '{DATASET_FILE_PATH.absolute().as_posix()}'.")
    
    def file_writer(output_path: Path, content: str) -> None:
        with open(output_path, "w") as file:
            file.write(content)
    
    result_saver = CsvResultSaver(file_writer)
    
    start_cli_menu(dataset, stats, recommender, evaluator, predictor, chosen_similarity, result_saver, RESULTS_PATH)

if __name__ == "__main__":
    main()