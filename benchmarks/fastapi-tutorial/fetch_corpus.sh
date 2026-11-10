#!/usr/bin/env bash
# Fetch the FastAPI tutorial docs (MIT licence) at a pinned commit into data/raw/.
# The gold quotes in questions.jsonl are resolved against exactly this revision.
set -euo pipefail

COMMIT=84d5f8ea759355e26d5a36f9c5c5652df924328e
DEST=${1:-data/raw/fastapi-tutorial}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git -C "$WORK" init -q
git -C "$WORK" remote add origin https://github.com/fastapi/fastapi.git
git -C "$WORK" sparse-checkout set --no-cone /docs/en/docs/tutorial/ /LICENSE
git -C "$WORK" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$WORK" checkout -q FETCH_HEAD

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$WORK/docs/en/docs/tutorial/." "$DEST/"
cp "$WORK/LICENSE" "$DEST/.LICENSE"
echo "FastAPI tutorial at $COMMIT -> $DEST ($(find "$DEST" -name '*.md' | wc -l) files)"
