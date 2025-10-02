#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .git ]]; then
  echo "[ERR] Not a git repository: $(pwd)"
  exit 2
fi

git fetch origin >/dev/null 2>&1 || true

LOCAL=$(git rev-parse @ 2>/dev/null || echo "")
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
BASE=$(git merge-base @ @{u} 2>/dev/null || echo "")

if [[ -z "$LOCAL" || -z "$REMOTE" || -z "$BASE" ]]; then
  echo "[ERR] No upstream tracking branch set (did you run 'git branch --set-upstream-to=origin/main'?)"
  exit 3
fi

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[OK] Repo is up-to-date with origin/main"
    exit 0
elif [ "$LOCAL" = "$BASE" ]; then
    echo "[PULL] Local is behind → run: git pull"
    exit 10
elif [ "$REMOTE" = "$BASE" ]; then
    echo "[PUSH] Local is ahead → run: git push"
    exit 11
else
    echo "[ERR] Local and origin/main diverged → manual fix needed"
    exit 20
fi