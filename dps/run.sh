#!/usr/bin/env -S bash --login
set -euo pipefail

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
repo_root=$(dirname $basedir)

MODEL_NAME=$1
START_YEAR=$2
END_YEAR=$3

conda run -n eis_model python $repo_root/model/main.py --model_name "${MODEL_NAME}" \
--start_year "${START_YEAR}" --end_year "${END_YEAR}"
