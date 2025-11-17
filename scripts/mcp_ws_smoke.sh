#!/usr/bin/env zsh
set -euo pipefail

# Heartbeat and streaming enabled
export LEXON_MCP_WS_ADDR=${LEXON_MCP_WS_ADDR:-127.0.0.1:8092}
export LEXON_MCP_HEARTBEAT_MS=${LEXON_MCP_HEARTBEAT_MS:-200}
export LEXON_MCP_STREAM=${LEXON_MCP_STREAM:-1}

echo "[start] WS server at $LEXON_MCP_WS_ADDR (heartbeat ${LEXON_MCP_HEARTBEAT_MS}ms, stream $LEXON_MCP_STREAM)"
./target/debug/lexc --mcp-ws &>/tmp/lexon_ws_server.log &
srv_pid=$!
sleep 0.3

echo "[client] Connecting and sending tool_call web.search"
export WS_URL="ws://${LEXON_MCP_WS_ADDR}"
cargo run -q -p lexc --bin ws_client || true

echo "[stop] Killing WS server ($srv_pid)"
kill $srv_pid 2>/dev/null || true
wait $srv_pid 2>/dev/null || true

echo "[done]"


