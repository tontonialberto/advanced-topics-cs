# Fairness in Group Recommendations - Assignments

Author: Alberto Tontoni

## Table of Contents
- [Description](#description)
- [How to use the application](#how-to-use-the-application)
    - [Build and Run](#build-and-run)
    - [Command Line Interface](#command-line-interface)
    - [Code Organization](#code-organization)


## Description

Group Recommendation System which uses the User-Based Collaborative Filtering approach.

A detailed explanation of the work done can be found here:
- [Assignment 1 Report](./reports/report-assignment1.md)

## How to use the application

The proposed Recommendation System has been realized as a containerized application which exposes a Command Line Interface. Experiments results can be either inspected inside the terminal or saved to CSV files.

### Build and Run

The following software needs to be installed on your machine:
- Docker and Docker Compose
- A Bash shell

The following commands assume that the repository is downloaded locally, and your terminal working directory is the assignment folder.

To build the application, launch the script `build-app.sh`.

To run the application, launch the script `run-app.sh [SIMILARITY_FUNC] [PREDICTION_FUNC]`:
- `SIMILARITY_FUNC` is an optional parameter to specify the similarity function to be used by the RS. Allowed values are `pearson` (default), `jaccard` and `itr`;
- `PREDICTION_FUN` is an optional parameter to specify the prediction function to be used by the RS. Allowed values are `mean_centered_abs` (default) and `mean_centered_no_abs` (the former is the Mean-Centered Aggregation presented above, the latter is the formula seen in class).


### Command Line Interface
Once launched, the application shows a menu like the following:

![](./resources/report-images/command-line-interface.png)

Commands from 1 to 4 allow to repeat the assignment tasks from A to E. 

Command 5, if selected, will prompt you to select a user and it will conduct all the assignment tasks on that user, also saving the results on different CSV files in the `results/assignment1` directory:
- Files named `most_relevant_10_items_for_user_<UserId>_<SimilarityName>.csv` contain the most relevant recommendations for user with id UserId using similarity SimilarityName;
- Files named `most_similar_10_users_for_user_<UserId>_<SimilarityName>.csv` contain the most similar users for a given user, using the given similarity;
- Files named `prediction_evaluation_user_<UserId>.csv` contain the result of the evaluation experiment mentioned above on the given user;
- Files named `user_similarity_matrix_<SimilarityName>.csv` contain the user similarity matrix of the MovieLens 100k dataset for the given similarity.

Command 6 computes the user similarity matrix for the similarity function chosen at application startup. The performances of the matrix computation have been hugely improved during development: the first implementation required approximately 30 minutes to compute PCC matrix on a small laptop, whereas now it takes from 4 to 10 seconds (you can take a look at the commit history to see how the Dataset class has been tweaked to precompute a lot of values). 

Commands 7 to 9 are just utilities.

### Code Organization

The project has been implemented using the Hexagonal Architecture, which promotes high testability. All the business logic is separated by the ways to interact with external systems. This means that the code could be easily extended to be exposed, for example, as a RESTful service instead of a CLI application.

The project source code lies in the `src.app` directory.
Here, the most interesting module is `domain`, which contains the following submodules:
- `prediction`: contains the prediction functions implementations;
- `similarity`: contains the prediction functions implementations.