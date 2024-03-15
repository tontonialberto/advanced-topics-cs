#!/bin/bash

SIMILARITY_FUNC=${1:-"pearson"}

if [[ "$SIMILARITY_FUNC" != "pearson" && "$SIMILARITY_FUNC" != "itr" && "$SIMILARITY_FUNC" != "jaccard" ]]; then
    echo "Invalid value for SIMILARITY_FUNC. Please provide either 'pearson', 'itr', or 'jaccard'."
    exit 1
fi

docker compose up --no-start
docker compose run -it -e SIMILARITY_FUNC="$SIMILARITY_FUNC" app