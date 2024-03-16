from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, UserId
from app.domain.recommender import PerformanceEvaluator, Prediction, Recommender, Stats
from app.domain.similarity.similarity import Similarity
from app.domain.utils import calculate_execution_time
from tabulate import tabulate


def start_cli_menu(dataset: Dataset, stats: Stats, recommender: Recommender, evaluator: PerformanceEvaluator, predictor: Prediction, similarity: Similarity) -> None:
    while True:
        print("")
        print("CLI Menu:")
        print("1) Display dataset information")
        print("2) Show 10 highest similarities for a selected user")
        print("3) Recommend 10 most relevant movies for a selected user")
        print("4) Evaluate predictions for a selected user, based on different similarity functions")
        print("5) Show the user similarity matrix")
        print("6) Show items rated by a selected user")
        print("7) Show commonly rated items between two users")
        print("8) Show similarity between two users")
        print("0) Exit")

        choice = input(">> ")

        if choice == "1":
            display_dataset_info(dataset)
        elif choice == "2":
            user_id = prompt_user_id()
            display_most_similar_users(user_id, limit=10, stats=stats)
        elif choice == "3":
            user_id = prompt_user_id()
            display_most_relevant_recommendations(user_id, limit=10, recommender=recommender)
        elif choice == "4":
            user_id = prompt_user_id()
            display_prediction_comparison(user_id, dataset, evaluator)
        elif choice == "5":
            display_user_similarity_matrix(stats)
        elif choice == "6":
            user_id = prompt_user_id()
            display_user_ratings(user_id, dataset)
        elif choice == "7":
            user_a = prompt_user_id()
            user_b = prompt_user_id()
            display_commonly_rated_items(user_a, user_b, dataset)
        elif choice == "8":
            user_a = prompt_user_id()
            user_b = prompt_user_id()
            display_similarity_between_two_users(user_a, user_b, similarity)
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")
            
def prompt_user_id() -> UserId: # type: ignore
    error = True
    while error:
        try:
            user_id = int(input("Enter user id: "))
            return user_id
        except ValueError:
            pass

def display_dataset_info(dataset: Dataset) -> None:
    first_rows = dataset.get_first(10)
    headers = ["User ID", "Item ID", "Rating"]
    table = [[user, item, rating] for user, item, rating in first_rows]
    
    print("")
    print(f"{len(dataset)} total rows.")
    print("")
    print("Displaying first 10 rows:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
@calculate_execution_time    
def display_most_similar_users(user: UserId, limit: int, stats: Stats) -> None:
    print("")
    print("Calculating...")
    most_similar_users = stats.get_most_similar_users(user, limit)
    headers = ["User", "Similarity"]
    table = [[user, f"{similarity:.8f}"] for user, similarity in most_similar_users]
    
    print(f"Most similar users to user {user} are:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))    
    
def display_user_ratings(user: UserId, dataset: Dataset) -> None:
    ratings = dataset.get_ratings_by_user(user)
    headers = ["Item", "Rating"]
    table = [[item, rating] for item, rating in ratings]
    
    print("")
    print(f"Ratings of user {user} ({len(ratings)} total):")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    print("")
    
@calculate_execution_time
def display_most_relevant_recommendations(user: UserId, limit: int, recommender: Recommender) -> None:
    print("")
    print("Calculating...")
    recommendations = recommender.get_recommendations(user, limit)
    headers = ["Item", "Predicted Rating"]
    table = [[item, predicted_rating] for item, predicted_rating in recommendations]
    
    print(f"Most relevant items for user {user}:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    print("")
    
def get_mean_absolute_error(comparisons: List[Tuple[float, float]]) -> float:
    return sum(abs(true_rating - predicted_rating) for true_rating, predicted_rating in comparisons) / len(comparisons)

@calculate_execution_time
def display_prediction_comparison(user: UserId, dataset: Dataset, evaluator: PerformanceEvaluator) -> None:
    print("")
    print("Calculating...")

    comparison = evaluator.get_comparison_by_user(user)
    predictors = evaluator.predictor_names
    headers = [
        "Item", 
        "True Rating", 
        *[f'Pred. ({predictor})' for predictor in predictors], 
        *[f'Abs. Error ({predictor})' for predictor in predictors], 
        "Best"
    ]
    table = []
    scores: Dict[str, float] = {predictor: 0 for predictor in predictors}
    mean_absolute_errors: Dict[str, float] = {predictor_name: evaluation.mean_absolute_error for predictor_name, evaluation in comparison.items()}
    
    for item, actual_rating in dataset.get_ratings_by_user(user):
        predicted_ratings = [evaluation.predictions[item].prediction for evaluation in comparison.values()]
        prediction_errors = [evaluation.predictions[item].absolute_error for evaluation in comparison.values()]
        best_predictor = predictors[prediction_errors.index(min(prediction_errors))]
        row = [
            item, 
            actual_rating, 
            *[f"{rating:.8f}" for rating in predicted_ratings], 
            *[f"{error:.8f}" for error in prediction_errors], 
            best_predictor
        ]
        table.append(row)
        scores[best_predictor] += 1
    
    print(f"Comparison of predictions for user {user}:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
    headers = ["Predictor", "Score", "Mean Absolute Error"]
    table = [[predictor, scores[predictor], mean_absolute_errors[predictor]] for predictor in predictors]
    
    print("")
    print("Scores:")
    print(tabulate(table, headers=headers, tablefmt="pretty"))

@calculate_execution_time
def compute_and_save_predictions(evaluator: PerformanceEvaluator, predictor: Prediction) -> None:
    print("")
    print("Calculating...")
    all_predictions = evaluator.get_all_predictions(predictor)
    print("Saving to file...")
    with open("predictions.csv", "w") as file:
        file.write("userId,movieId,rating\n")
        for user, predictions in all_predictions.items():
            for item, rating in predictions:
                file.write(f"{user},{item},{rating}\n")
    print("Done.")
    print("")
    
@calculate_execution_time
def display_user_similarity_matrix(stats: Stats) -> None:
    print("")
    print("Calculating...")
    user_similarity_matrix = stats.get_user_similarity_matrix()
    
    print("Displaying user similarity matrix:")
    print("")
    print(tabulate(user_similarity_matrix.items(), headers=["Users", "Similarity"], tablefmt="pretty"))
    print("")
    
@calculate_execution_time
def display_similarity_between_two_users(user_a: UserId, user_b: UserId, similarity: Similarity) -> None:
    sim = similarity.get_similarity(user_a, user_b)
    print("")
    print(f"Similarity between user {user_a} and user {user_b} is: {sim:.8f}")
    print("")
    
def display_commonly_rated_items(user_a: UserId, user_b: UserId, dataset: Dataset) -> None:
    common_items = dataset.get_items_rated_by_both(user_a, user_b)
    headers = ["Item", f"Rating of User {user_a}", f"Rating of User {user_b}"]
    table = [[item, rating_a, rating_b] for item, (rating_a, rating_b) in common_items.items()]
    
    print("")
    print(f"Items rated by both user {user_a} and user {user_b} ({len(common_items)} total):")
    print("")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    print("")
