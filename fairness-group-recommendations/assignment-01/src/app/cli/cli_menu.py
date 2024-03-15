from app.domain.dataset import Dataset, UserId
from app.domain.recommender import PerformanceEvaluator, Recommender, Stats
from app.domain.utils import calculate_execution_time


def start_cli_menu(dataset: Dataset, stats: Stats, recommender: Recommender, evaluator: PerformanceEvaluator) -> None:
    while True:
        print("")
        print("CLI Menu:")
        print("1) Display dataset information")
        print("2) Show 10 highest similarities for a selected user")
        print("3) Recommend 10 most relevant movies for a selected user")
        print("4) Evaluate performance of two predictors, based on different similarity functions (Pearson and ITR), on a selected user's ratings")
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
            display_prediction_comparison(user_id, evaluator)
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")
            
def prompt_user_id() -> UserId:
    error = True
    while error:
        try:
            user_id = int(input("Enter user id: "))
            return user_id
        except ValueError:
            pass

def display_dataset_info(dataset: Dataset) -> None:
    print("")
    print(f"{len(dataset.data())} total rows.")
    print("")
    print("Displaying first 10 rows:")
    first_rows = dataset.data()[:10]
    for row in first_rows:
        print(row)
    print("")

@calculate_execution_time    
def display_most_similar_users(user: UserId, limit: int, stats: Stats) -> None:
    print("")
    print("Calculating...")
    most_similar_users = stats.get_most_similar_users(user, limit)
    print(f"Most similar users to user {user} are:")
    print("")
    print("User\tSimilarity")
    print("----\t----------")
    for user, similarity in most_similar_users:
        print(f"{user}\t{similarity}")
    print("")
    
def display_user_ratings(user: UserId, dataset: Dataset) -> None:
    ratings = dataset.get_ratings_by_user(user)
    print("")
    print(f"Ratings of user {user} ({len(ratings)} total):")
    for item, rating in ratings:
        print(f"{item}\t{rating}")
    print("")
    
@calculate_execution_time
def display_most_relevant_recommendations(user: UserId, limit: int, recommender: Recommender) -> None:
    print("")
    print("Calculating...")
    recommendations = recommender.get_recommendations(user, limit)
    print(f"Most relevant items for user {user}:")
    print("Item\tPredicted Rating")
    print("----\t----------------")
    for item, predicted_rating in recommendations:
        print(f"{item}\t{predicted_rating}")
    print("")

@calculate_execution_time
def display_prediction_comparison(user: UserId, evaluator: PerformanceEvaluator) -> None:
    print("")
    print("Calculating...")
    comparison = evaluator.get_comparison_by_user(user)
    predictors = evaluator.predictor_names
    print(f"Comparison of predictions for user {user}:")
    print(f"Item\tRating\t" + "\t".join([f"Pred. ({predictor})" for predictor in predictors]) + "\t" + "\t".join([f"Error ({predictor})" for predictor in predictors]) + "\tBest")
    print("----\t------\t" + '\t'.join(['------------' for predictor in predictors]) + "\t" + '\t'.join(['-------------' for predictor in predictors]) + "\t----")
    for item, comparison in comparison.items():
        best_predictor = predictors[comparison.errors.index(min(comparison.errors))]
        print(f"{item}\t{comparison.actual_rating}\t" + "\t".join([f"{predicted_rating:.8f}" for predicted_rating in comparison.predicted_ratings]) + "\t" + "\t".join([f"{error:.8f}" for error in comparison.errors]) + f"\t{best_predictor}")
    print("")