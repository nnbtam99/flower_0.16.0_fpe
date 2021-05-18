#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

python -m isort --check-only flwr_baselines         && echo "- isort:  done" &&
python -m black --check flwr_baselines              && echo "- black:  done" &&
python -m mypy flwr_baselines                       && echo "- mypy:   done" &&
python -m pylint flwr_baselines                     && echo "- pylint: done" &&
python -m pytest --durations=0 -v flwr_baselines    && echo "- pytest: done" &&
echo "- All Python checks passed"
