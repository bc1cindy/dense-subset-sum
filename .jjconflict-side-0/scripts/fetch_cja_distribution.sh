#!/usr/bin/env bash
# Download the CJA empirical Bitcoin UTXO distribution (Git LFS, ~353 MB).
#
# This is the reference empirical distribution for sampling CoinJoin-like
# input sets. It is stored in the payjoin/cja repo via Git LFS,
# so a normal `git clone` of that repo only pulls a 134-byte pointer. This
# script fetches the actual content via GitHub's LFS media URL.
#
# Output: testdata/cja_distribution.bin
# Expected SHA-256: 2ee40b0ec7d24678a1e8a15bcd65b564339725daec39a1cd3c9a2d4c3438ff23
# Expected size:    370,615,727 bytes
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/testdata"
OUT_FILE="$OUT_DIR/cja_distribution.bin"
URL="https://media.githubusercontent.com/media/payjoin/cja/master/distribution.bin"
EXPECTED_SHA="2ee40b0ec7d24678a1e8a15bcd65b564339725daec39a1cd3c9a2d4c3438ff23"
EXPECTED_SIZE=370615727

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_FILE" ]]; then
    actual_size=$(stat -f%z "$OUT_FILE" 2>/dev/null || stat -c%s "$OUT_FILE")
    if [[ "$actual_size" == "$EXPECTED_SIZE" ]]; then
        echo "Already present: $OUT_FILE ($actual_size bytes)"
        exit 0
    fi
    echo "File exists but size mismatch ($actual_size != $EXPECTED_SIZE), redownloading..."
fi

echo "Downloading CJA distribution.bin (~353 MB) to $OUT_FILE"
curl -L --fail --progress-bar -o "$OUT_FILE" "$URL"

actual_size=$(stat -f%z "$OUT_FILE" 2>/dev/null || stat -c%s "$OUT_FILE")
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
    echo "ERROR: size mismatch — got $actual_size, expected $EXPECTED_SIZE" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha=$(sha256sum "$OUT_FILE" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    actual_sha=$(shasum -a 256 "$OUT_FILE" | awk '{print $1}')
else
    echo "No sha256 tool found; skipping checksum verification."
    actual_sha=""
fi

if [[ -n "$actual_sha" && "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: sha256 mismatch — got $actual_sha, expected $EXPECTED_SHA" >&2
    exit 1
fi

echo "OK: $OUT_FILE ($actual_size bytes, sha256 verified)"
