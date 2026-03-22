#!/bin/bash

# Clean up old Cloud Run job executions in the i4g-ml project.
# Mirrors core/scripts/infra/clean_cloud_run_history.sh for the ML project.

region="us-central1"
project="i4g-ml"
jobs="etl-ingest"

for job in $jobs; do
    gcloud run jobs executions list \
        --job=$job \
        --project=$project \
        --region=$region \
        --format=json 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin) if sys.stdin.readable() else []
for ex in data:
    ct = ex.get('completionTime') or (ex.get('status') or {}).get('completionTime')
    name = (ex.get('metadata') or {}).get('name') or ex.get('name', '').rsplit('/', 1)[-1]
    if ct and name:
        print(name)
" | xargs -r -I {} gcloud run jobs executions delete {} \
        --project=$project \
        --region=$region \
        --quiet
done
