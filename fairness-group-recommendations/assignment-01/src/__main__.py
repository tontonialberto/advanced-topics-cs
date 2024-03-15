from pathlib import Path
import os
from typing import Dict, Tuple
from app.cli.cli_menu import start_cli_menu
from app.data_loader.file_data_loader import FileDataLoader
from app.domain.dataset import Dataset
from app.domain.recommender import PerformanceEvaluator, PredictionImpl, Recommender, Stats
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
    
    print(f"Using similarity function: {SIMILARITY_FUNC}")
    
    loader = FileDataLoader(DATASET_FILE_PATH)
    dataset = loader.load()
    
    similarity_functions = create_similarity_functions(dataset)
    chosen_similarity = similarity_functions[SIMILARITY_FUNC]

    stats = Stats(dataset, chosen_similarity)
    predictor = PredictionImpl(dataset, chosen_similarity)
    recommender = Recommender(dataset, predictor)
    
    predictors_for_comparison = [
        (name, PredictionImpl(dataset, similarity))
        for name, similarity in similarity_functions.items()
    ]
    
    # predictor_for_comparison = PredictionImpl(dataset, similarity=CachedSimilarity(
    #     ITR(dataset) if SIMILARITY_FUNC == "pearson" else PearsonCorrelation(dataset)
    # ))
    # if SIMILARITY_FUNC == "pearson":
    #     predictor_name = "Pearson"
    #     predictor_for_comparison_name = "ITR"
    # else:
    #     predictor_name = "ITR"
    #     predictor_for_comparison_name = "Pearson"
    evaluator = PerformanceEvaluator(predictors_for_comparison, dataset)
    
    print(f"Loaded '{DATASET_FILE_PATH.absolute().as_posix()}'.")
    
    start_cli_menu(dataset, stats, recommender, evaluator)

if __name__ == "__main__":
    main()