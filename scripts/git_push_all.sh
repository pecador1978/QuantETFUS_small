#!/usr/bin/env bash
set -euo pipefail
REPO="/Users/Finance/QuantETFUS_small"
cd "$REPO"
git fetch origin >/dev/null 2>&1 || true
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  git branch --set-upstream-to=origin/main main
fi
git add .
MSG="auto: $(date -u +'%Y-%m-%d %H:%M UTC') sync from Mini"
git commit -m "$MSG" || true
git push
echo "[OK] Pushed to origin/main"
