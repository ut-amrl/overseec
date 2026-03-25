#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-image-edit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements-image-edit-test.txt"

export OVERSEEC_IMAGE_EDIT_TEST_MODE=1
python "$ROOT_DIR/scripts/create_image_edit_test_fixtures.py"

cat <<EOF

Image editing test environment is ready.

Next steps:
  1. source "$VENV_DIR/bin/activate"
  2. export OVERSEEC_IMAGE_EDIT_TEST_MODE=1
  3. cd "$ROOT_DIR/frontend"
  4. python app.py

Then open http://127.0.0.1:5002 and use:
  - TIFF Browse -> select "edit_test_scene"
  - Masks Browse -> edit_test_scene / mask / temp_latest
  - Costmap Browse -> edit_test_scene / costmap / temp_latest

EOF
