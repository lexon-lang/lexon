#!/usr/bin/env zsh
set -euo pipefail

export LEXON_MCP_HEARTBEAT_MS=${LEXON_MCP_HEARTBEAT_MS:-0}
export LEXON_MCP_RATE_PER_MIN=${LEXON_MCP_RATE_PER_MIN:-5}
export LEXON_MCP_AUTH_BEARER=${LEXON_MCP_AUTH_BEARER:-secret}

echo "[start] stdio server (oneshot)"
(
  ./target/debug/lexc --mcp-stdio << 'EOF'
{"jsonrpc":"2.0","id":1,"method":"list_tools"}
{"jsonrpc":"2.0","id":2,"method":"auth","params":{"bearer":"secret"}}
{"jsonrpc":"2.0","id":3,"method":"list_tools"}
{"jsonrpc":"2.0","id":4,"method":"tool_call","params":{"name":"read_file","args":{"path":"README.md"}}}
exit
EOF
) | sed -n '1,80p'

echo "[done]"


