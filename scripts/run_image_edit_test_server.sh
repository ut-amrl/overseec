#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-image-edit}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv at $VENV_DIR. Run scripts/setup_image_edit_test_env.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export OVERSEEC_IMAGE_EDIT_TEST_MODE=1

python "$ROOT_DIR/scripts/create_image_edit_test_fixtures.py"

cd "$ROOT_DIR/frontend"
python app.py
