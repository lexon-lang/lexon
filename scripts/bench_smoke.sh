#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

samples=(
  "samples/00-hello-lexon.lx"
  "samples/streaming/multioutput.lx"
  "samples/apps/research_analyst/main.lx"
)

out="/tmp/lexon_bench.json"
echo "{" > "$out"
first=1
for s in "${samples[@]}"; do
  t0=$(date +%s%3N)
  cargo run -q -p lexc-cli -- compile --run "$s" >/dev/null 2>&1 || true
  t1=$(date +%s%3N)
  delta=$((t1 - t0))
  if [[ $first -eq 0 ]]; then echo "," >> "$out"; fi
  echo "\"$s\": $delta" >> "$out"
  first=0
done
echo "}" >> "$out"
echo "$out"


