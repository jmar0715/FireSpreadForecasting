#!/usr/bin/env -S bash --login
set -euo pipefail

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
repo_root=$(dirname $basedir)

MODEL_NAME=$1
TRAINING_SET=$2
VALIDATION_SET=$3

python $repo_root/model/main.py --model_name "${MODEL_NAME}" \
--train_data "${TRAINING_SET}" --validation_data "${VALIDATION_SET}"