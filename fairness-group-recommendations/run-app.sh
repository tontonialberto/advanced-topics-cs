#!/bin/bash

SIMILARITY_FUNC=${1:-"pearson"}

if [[ "$SIMILARITY_FUNC" != "pearson" && "$SIMILARITY_FUNC" != "itr" && "$SIMILARITY_FUNC" != "jaccard" ]]; then
    echo "Invalid value for SIMILARITY_FUNC. Please provide either 'pearson', 'itr', or 'jaccard'."
    exit 1
fi

PREDICTION_FUNC=${2:-"mean_centered_abs"}

if [[ "$PREDICTION_FUNC" != "mean_centered_abs" && "$PREDICTION_FUNC" != "mean_centered_no_abs" ]]; then
    echo "Invalid value for PREDICTION_FUNC. Please provide either 'mean_centered_abs' or 'mean_centered_no_abs'."
    exit 1
fi

docker compose up --no-start
docker compose run -it -e SIMILARITY_FUNC="$SIMILARITY_FUNC" -e PREDICTION_FUNC="$PREDICTION_FUNC" app