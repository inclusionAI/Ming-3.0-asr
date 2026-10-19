#!/usr/bin/env bash
set -euo pipefail
: "${SGLANG_DIR:?Set SGLANG_DIR to the SGLang source checkout}"
GUIDE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_COMMIT=81f27fb3a71d3504eab821a9d39352c924c730cf
ADAPTER_COMMIT=bba862771a9e3413a062dad33e8b494ea9f7755b
ALIAS_PATCH="$GUIDE_DIR/patches/ming-v3-asr-alias.patch"

if git -C "$SGLANG_DIR" apply --reverse --check "$ALIAS_PATCH" 2>/dev/null; then
  echo "Ming ASR patches are already applied."
  exit 0
fi

commit="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
case "$commit" in
  "$BASE_COMMIT")
    for name in \
      0001-Model-Support-Ming-ASR-Tiny-3.0.patch \
      0002-Model-Fix-Ming-audio-RoPE-and-typed-config-overrides.patch; do
      patch="$GUIDE_DIR/patches/$name"
      if ! git -C "$SGLANG_DIR" apply --check "$patch"; then
        echo "Patch does not match this checkout: $name. Start from a clean $BASE_COMMIT worktree." >&2
        exit 1
      fi
      git -C "$SGLANG_DIR" apply "$patch"
      echo "Applied: $name"
    done
    ;;
  "$ADAPTER_COMMIT")
    echo "Ming ASR support is already present in this adapter commit."
    ;;
  *)
    echo "Expected upstream $BASE_COMMIT or adapter $ADAPTER_COMMIT; found $commit." >&2
    exit 1
    ;;
esac

if ! git -C "$SGLANG_DIR" apply --check "$ALIAS_PATCH"; then
  echo "The MingV3ASR alias patch does not match this checkout." >&2
  exit 1
fi
git -C "$SGLANG_DIR" apply "$ALIAS_PATCH"
echo "Applied: ming-v3-asr-alias.patch"
