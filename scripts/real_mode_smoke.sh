#!/usr/bin/env zsh
set -euo pipefail

# Real-mode smoke for LLM + Web Search
# Requirements:
#   - Set provider key and model, e.g.:
#       export OPENAI_API_KEY=...
#       export LEXON_MODEL=openai:gpt-4o-mini
#   - Optional: real web search endpoint
#       export LEXON_WEB_SEARCH_ENDPOINT="https://your.search.service/search"
#   - Optional: HTTP loaders
#       export LEXON_ALLOW_HTTP=1
#       export LEXON_HTTP_TIMEOUT_MS=10000
#       export LEXON_HTTP_RETRIES=2

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT_DIR/target/debug/lexc-cli"

# Load user shell env if present (to pick up OPENAI_API_KEY, etc.)
if [[ -f "$HOME/.zshrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.zshrc"
fi
if [[ -f "$HOME/.zprofile" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.zprofile"
fi

echo "[real-smoke] Building..."
pushd "$ROOT_DIR" >/dev/null
cargo build -q -p lexc -p lexc-cli
popd >/dev/null

MODEL=${LEXON_MODEL:-}
if [[ -z "${MODEL}" ]]; then
  echo "[real-smoke] LEXON_MODEL not set. Defaulting to openai:gpt-4o-mini"
  export LEXON_MODEL=openai:gpt-4o-mini
fi

PROVIDER=${LEXON_MODEL%%:*}

case "$PROVIDER" in
  openai)
    : "${OPENAI_API_KEY?Missing OPENAI_API_KEY for OpenAI real-mode.}" ;;
  anthropic)
    : "${ANTHROPIC_API_KEY?Missing ANTHROPIC_API_KEY for Anthropic real-mode.}" ;;
  google)
    : "${GOOGLE_API_KEY?Missing GOOGLE_API_KEY for Google real-mode.}" ;;
  *)
    echo "[real-smoke] Unknown provider in LEXON_MODEL: '$PROVIDER'" >&2
    exit 2
    ;;
esac

RA_SAMPLE="$ROOT_DIR/samples/apps/research_analyst/main.lx"
WEB_SMOKE="$ROOT_DIR/samples/web/web_search_smoke.lx"

echo "[real-smoke] Running Research Analyst with real provider: $LEXON_MODEL"
set +e
RA_OUT=$("$BIN" compile --run "$RA_SAMPLE" 2>&1)
rc=$?
set -e
echo "$RA_OUT" > /tmp/ra_real_run.log
if [[ $rc -ne 0 ]]; then
  echo "[real-smoke] Research Analyst FAILED (exit $rc)" >&2
  exit $rc
fi

if echo "$RA_OUT" | grep -qi "falling back to simulated"; then
  echo "[real-smoke] Detected fallback to simulated in Research Analyst output." >&2
  exit 3
fi

echo "[real-smoke] Running web search smoke"
set +e
WEB_OUT=$("$BIN" compile --run "$WEB_SMOKE" 2>&1)
rc=$?
set -e
echo "$WEB_OUT" > /tmp/web_real_run.log
if [[ $rc -ne 0 ]]; then
  echo "[real-smoke] Web search smoke FAILED (exit $rc)" >&2
  exit $rc
fi

if [[ -z "${LEXON_WEB_SEARCH_ENDPOINT:-}" ]]; then
  echo "[real-smoke] LEXON_WEB_SEARCH_ENDPOINT not set; search ran in simulated mode by design."
else
  if echo "$WEB_OUT" | grep -q "SIMULATED SEARCH RESULT"; then
    echo "[real-smoke] Web search appears simulated despite endpoint being set." >&2
    exit 4
  fi
fi

echo "[real-smoke] SUCCESS: real-mode checks passed. Logs: /tmp/ra_real_run.log, /tmp/web_real_run.log"

