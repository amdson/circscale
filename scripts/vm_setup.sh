#!/usr/bin/env bash
# One-shot setup for a fresh GPU VM (tested against Lambda's Ubuntu images).
# Usage: git clone <repo> && cd <repo> && bash scripts/vm_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra cuda
uv run python -c "import jax; print('jax backend:', jax.default_backend(), jax.devices())"
uv run pytest -q

echo
echo "setup complete. next:"
echo "  tmux new -s grid"
echo "  uv run python sgd_grid.py all      # tune -> grid -> figs (resumable)"
echo "  uv run python sgd_grid.py status   # progress"
