#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/vllmtop.py"
BIN_DIR="$HOME/.local/bin"
DEST="$BIN_DIR/vllmtop"

if [[ ! -f "$SRC" ]]; then
    echo "install.sh: $SRC not found" >&2
    exit 1
fi

mkdir -p "$BIN_DIR"
chmod +x "$SRC"
ln -sfn "$SRC" "$DEST"

echo "linked $DEST -> $SRC"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "warning: $BIN_DIR is not in PATH; add it to your shell rc to use 'vllmtop' from anywhere" >&2 ;;
esac
