#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .git ]]; then
  echo "[ERR] Not a git repository: $(pwd)"
  exit 1
fi

git fetch origin >/dev/null 2>&1

LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})
BASE=$(git merge-base @ @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[OK] Repo is up-to-date with origin/main"
elif [ "$LOCAL" = "$BASE" ]; then
    echo "[PULL] Local is behind → run: git pull"
elif [ "$REMOTE" = "$BASE" ]; then
    echo "[PUSH] Local is ahead → run: git push"
else
    echo "[ERR] Local and origin/main diverged → manual fix needed"
fi