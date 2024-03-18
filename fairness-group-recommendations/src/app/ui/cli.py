import os
from pathlib import Path
from typing import Dict, List, Tuple
from app.domain.dataset import Dataset, ItemId, UserId
from app.domain.group_prediction.disagreement.disagreement import Disagreement
from app.domain.group_prediction.group_prediction import Group
from app.domain.group_recommender import GroupRecommender
from app.domain.recommender import Evaluation, PerformanceEvaluator, Prediction, PredictorName, Recommender, Stats
from app.domain.result_saver import ResultSaver
from app.domain.similarity.similarity import Similarity
from app.domain.utils import calculate_execution_time
from tabulate import tabulate


TABLE_RESULTS_LIMIT = 10

def start_cli_menu(
        dataset: Dataset, 
        stats: Stats, 
        recommender: Recommender, 
        evaluator: PerformanceEvaluator, 
        predictor: Prediction, 
        similarity: Similarity, 
        result_saver: ResultSaver, 
        results_output_path: Path, 
        recommender_avg: GroupRecommender,
        recommender_least_misery: GroupRecommender,
        recommender_consensus: GroupRecommender,
        disagreement: Disagreement) -> None:
    
    global TABLE_RESULTS_LIMIT
    
    while True:
        print("")
        print("CLI Menu:")
        print(" Assignment 1:")
        print("  1) Display dataset information")
        print("  2) Show", TABLE_RESULTS_LIMIT, "highest similarities for a selected user")
        print("  3) Recommend", TABLE_RESULTS_LIMIT, "most relevant movies for a selected user")
        print("  4) Evaluate predictions for a selected user, based on different similarity functions")
        print("  5) Save results of assignment 1 to CSV files")
        print(" Assignment 2:")
        print("  6) Recommend", TABLE_RESULTS_LIMIT, "most relevant movies for a group of 3 users (Average Aggregation)")
        print("  7) Recommend", TABLE_RESULTS_LIMIT, "most relevant movies for a group of 3 users (Least Misery Aggregation)")
        print("  8) Recommend", TABLE_RESULTS_LIMIT, "most relevant movies for a group of 3 users (Average with Consensus based on Pairwise Disagremeents)")
        print("  9) Show average pairwise disagreement between a group of 3 users")
        print("  10) Show disagreements for all items in a group of 3 users")
        print(" Utilities:")
        print("  101) Compute the user similarity matrix")
        print("  102) Show items rated by a selected user")
        print("  103) Show commonly rated items between two users")
        print("  104) Show similarity between two users")
        print("  105) Predict rating for a user on an item")
        print(" System Options:")
        print("  201) Change table rows limit (default 10)")
        print("  202) Display system variables")
        print(" 0) Exit")

        choice = input(">> ")

        if choice == "1":
            display_dataset_info(dataset)
        elif choice == "2":
            user_id = prompt_user_id()
            display_most_similar_users(user_id, limit=TABLE_RESULTS_LIMIT, stats=stats)
        elif choice == "3":
            user_id = prompt_user_id()
            display_most_relevant_recommendations(user_id, limit=TABLE_RESULTS_LIMIT, recommender=recommender)
        elif choice == "4":
            user_id = prompt_user_id()
            display_prediction_comparison(user_id, dataset, evaluator)
        elif choice == "5":
            user = prompt_user_id()
            assignment1_output_folder = results_output_path / "assignment1"
            save_assigment1_results(user, assignment1_output_folder, result_saver, similarity, stats, recommender, evaluator, dataset)
        elif choice == "6":
            group = [prompt_user_id() for _ in range(3)]
            display_most_relevant_group_recommendations(group, limit=TABLE_RESULTS_LIMIT, recommender=recommender_avg, user_predictor=predictor, disagreement=disagreement)
        elif choice == "7":
            group = [prompt_user_id() for _ in range(3)]
            display_most_relevant_group_recommendations(group, limit=TABLE_RESULTS_LIMIT, recommender=recommender_least_misery, user_predictor=predictor, disagreement=disagreement)
        elif choice == "8":
            group = [prompt_user_id() for _ in range(3)]
            display_most_relevant_group_recommendations(group, limit=TABLE_RESULTS_LIMIT, recommender=recommender_consensus, user_predictor=predictor, disagreement=disagreement)
        elif choice == "9":
            group = [prompt_user_id() for _ in range(3)]
            item = prompt_integer("Enter item id: ")
            display_disagreements(group, [item], disagreement, limit=1)
        elif choice == "10":
            group = [prompt_user_id() for _ in range(3)]
            display_disagreements(group, dataset.get_all_items(), disagreement, limit=TABLE_RESULTS_LIMIT)
        elif choice == "101":
            compute_user_similarity_matrix(stats)
        elif choice == "102":
            user_id = prompt_user_id()
            display_user_ratings(user_id, dataset)
        elif choice == "103":
            user_a = prompt_user_id()
            user_b = prompt_user_id()
            display_commonly_rated_items(user_a, user_b, dataset)
        elif choice == "104":
            user_a = prompt_user_id()
            user_b = prompt_user_id()
            display_similarity_between_two_users(user_a, user_b, similarity)
        elif choice == "105":
            display_prediction(user_id=prompt_user_id(), item_id=prompt_integer("Enter item id: "), predictor=predictor)
        elif choice == "201":
            TABLE_RESULTS_LIMIT = prompt_integer("Enter new limit: ")
            print(f"Table results limit correctly set to {TABLE_RESULTS_LIMIT}.")
            print("")
        elif choice == "202":
            print(os.environ.items())
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")
            
def prompt_user_id() -> UserId:
    return prompt_integer("Enter user id: ")
            
def prompt_integer(message: str) -> UserId: # type: ignore
    error = True
    while error:
        try:
            user_id = int(input(message))
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
    print(tabulate(table, headers=headers, tablefmt="github"))
    
@calculate_execution_time    
def display_most_similar_users(user: UserId, limit: int, stats: Stats) -> None:
    print("")
    print("Calculating...")
    most_similar_users = stats.get_most_similar_users(user, limit)
    headers = ["User", "Similarity"]
    table = [[user, f"{similarity:.8f}"] for user, similarity in most_similar_users]
    
    print(f"Most similar users to user {user} are:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="github"))    
    
def display_user_ratings(user: UserId, dataset: Dataset) -> None:
    ratings = dataset.get_ratings_by_user(user)
    headers = ["Item", "Rating"]
    table = [[item, rating] for item, rating in ratings]
    
    print("")
    print(f"Ratings of user {user} ({len(ratings)} total):")
    print("")
    print(tabulate(table, headers=headers, tablefmt="github"))
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
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("")
    
def get_evaluation_tabular_output(user_items: List[Tuple[ItemId, float]], evaluation: Dict[PredictorName, Evaluation]) -> Tuple[List[str], List[List[str]]]:
    predictors = list(evaluation.keys())
    headers = [
        "Item", 
        "True Rating", 
        *[f'Pred. ({predictor})' for predictor in predictors], 
        *[f'Abs. Error ({predictor})' for predictor in predictors], 
        "Best"
    ]
    table = []
    for item, actual_rating in user_items:
        predicted_ratings = [evaluation.predictions[item].prediction for evaluation in evaluation.values()]
        prediction_errors = [evaluation.predictions[item].absolute_error for evaluation in evaluation.values()]
        best_predictor = predictors[prediction_errors.index(min(prediction_errors))]
        row = [
            str(item), 
            str(actual_rating), 
            *[f"{rating:.8f}" for rating in predicted_ratings], 
            *[f"{error:.8f}" for error in prediction_errors], 
            best_predictor
        ]
        table.append(row)
    return headers, table

@calculate_execution_time
def display_prediction_comparison(user: UserId, dataset: Dataset, evaluator: PerformanceEvaluator) -> None:
    print("")
    print("Calculating...")

    comparison = evaluator.get_comparison_by_user(user)
    predictors = evaluator.predictor_names
    user_items = dataset.get_ratings_by_user(user)
    headers, table = get_evaluation_tabular_output(user_items, comparison)
    scores: Dict[str, float] = {predictor: 0 for predictor in predictors}
    mean_absolute_errors: Dict[str, float] = {predictor_name: evaluation.mean_absolute_error for predictor_name, evaluation in comparison.items()}
    
    for item, _ in dataset.get_ratings_by_user(user):
        prediction_errors = [evaluation.predictions[item].absolute_error for evaluation in comparison.values()]
        best_predictor = predictors[prediction_errors.index(min(prediction_errors))]
        scores[best_predictor] += 1
    
    print(f"Comparison of predictions for user {user}:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="github"))
    
    headers = ["Predictor", "Score", "Mean Absolute Error"]
    table = [[predictor, scores[predictor], mean_absolute_errors[predictor]] for predictor in predictors]
    
    print("")
    print(f"Scores on user {user}:")
    print(tabulate(table, headers=headers, tablefmt="github"))

@calculate_execution_time
def compute_user_similarity_matrix(stats: Stats) -> None:
    print("")
    print("Calculating...")
    user_similarity_matrix = stats.get_user_similarity_matrix()
    
    print("User similarity matrix computed successfully.")
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
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("")

@calculate_execution_time
def save_assigment1_results(user: UserId, output_folder: Path, result_saver: ResultSaver, similarity: Similarity, stats: Stats, recommender: Recommender, evaluator: PerformanceEvaluator, dataset: Dataset) -> None:
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    
    print("")
    print("Computing similarity matrix...")
    similarity_matrix = stats.get_user_similarity_matrix()
    user_similarity_matrix_filepath = output_folder / f"user_similarity_matrix_{similarity.name}.csv"
    result_saver.save(
        output_path=user_similarity_matrix_filepath,
        headers=["userIdA", "userIdB", "similarity"],
        rows=[
            [str(user_a), str(user_b), f"{similarity:.8f}"]
            for (user_a, user_b), similarity in similarity_matrix.items()
        ],
    )
    print("Done.")
    
    print("")
    print(f"Computing highest 10 similarities for user {user}...")
    most_similar_users = stats.get_most_similar_users(user, limit=10)
    most_similar_users_filepath = output_folder / f"most_similar_10_users_for_user_{user}_{similarity.name}.csv"
    result_saver.save(
        output_path=most_similar_users_filepath,
        headers=["userId", "similarity"],
        rows=[
            [str(user_id), f"{similarity:.8f}"]
            for user_id, similarity in most_similar_users
        ],
    )
    print("Done.")
    
    print("")
    print(f"Computing the 10 most relevant items for user {user}...")
    most_relevant_recommendations = recommender.get_recommendations(user, limit=10)
    most_relevant_recommendations_filepath = output_folder / f"most_relevant_10_items_for_user_{user}_{similarity.name}.csv"
    result_saver.save(
        output_path=most_relevant_recommendations_filepath,
        headers=["itemId", "prediction"],
        rows=[
            [str(item), f"{predicted_rating:.8f}"]
            for item, predicted_rating in most_relevant_recommendations
        ],
    )
    print("Done.")
    
    print("")
    print(f"Evaluating predictions using different similarity functions for user {user}...")
    comparison = evaluator.get_comparison_by_user(user)
    prediction_evaluation_filepath = output_folder / f"prediction_evaluation_user_{user}.csv"
    headers, table = get_evaluation_tabular_output(dataset.get_ratings_by_user(user), comparison)
    result_saver.save(
        output_path=prediction_evaluation_filepath,
        headers=headers,
        rows=table,
    )
    print("Done.")
    
    print("")
    print("All done:")
    print(f"- User similarity matrix saved to: {user_similarity_matrix_filepath.as_posix()}")
    print(f"- Highest 10 similarities saved to: {most_similar_users_filepath.as_posix()}")
    print(f"- 10 most relevant items saved to: {most_relevant_recommendations_filepath.as_posix()}")
    print(f"- Prediction evaluation saved to: {prediction_evaluation_filepath.as_posix()}")
    
@calculate_execution_time
def display_most_relevant_group_recommendations(group: Group, limit: int, recommender: GroupRecommender, user_predictor: Prediction, disagreement: Disagreement) -> None:
    print("")
    print("Calculating...")
    recommendations = recommender.get_recommendations(group, limit)
    headers = ["#", "Item", "Pred. (Group)", *[f"Pred. (User {user})" for user in group], "Disagreement"]
    table = [
        [
            f"{index + 1}.",
            item, 
            group_prediction, 
            *[user_predictor.get_prediction(user, item) for user in group],
            disagreement.get_disagreement(group, item)
        ] 
        for index, (item, group_prediction) in enumerate(recommendations)
    ]
    
    print(f"Most relevant items for group {group}:")
    print("")
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("")

@calculate_execution_time
def display_disagreements(group: Group, item: List[ItemId], disagreement: Disagreement, limit: int) -> None:
    print("")
    print("Calculating...")
    
    disagreements: List[Tuple[ItemId, float]] = [
        (item, disagreement.get_disagreement(group, item))
        for item in item
    ]
    disagreements.sort(key=lambda x: x[1], reverse=True)
    disagreements = disagreements[:limit]
    
    headers = ["Item", "Disagreement"]
    table = [[item, disagreement] for item, disagreement in disagreements]
    print(f"Disagreements for group {group}:")
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("")
    
def display_prediction(user_id: UserId, item_id: ItemId, predictor: Prediction) -> None:
    prediction = predictor.get_prediction(user_id, item_id)
    print("")
    print(f"Predicted rating for user {user_id} on item {item_id}: {prediction:.8f}")
    print("")