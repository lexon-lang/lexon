#!/usr/bin/env zsh
set -euo pipefail

export LEXON_MCP_HEARTBEAT_MS=${LEXON_MCP_HEARTBEAT_MS:-0}
export LEXON_MCP_RATE_PER_MIN=${LEXON_MCP_RATE_PER_MIN:-10}
export LEXON_MCP_AUTH_BEARER=${LEXON_MCP_AUTH_BEARER:-secret}
export LEXON_MCP_ACL_FILE=${LEXON_MCP_ACL_FILE:-samples/mcp/acl.json}

echo "[start] stdio server with ACL (oneshot)"
(
  ./target/debug/lexc --mcp-stdio << 'EOF'
{"jsonrpc":"2.0","id":1,"method":"auth","params":{"bearer":"secret"}}
{"jsonrpc":"2.0","id":2,"method":"tool_call","params":{"name":"write_file","args":{"path":"output/tmp.txt","content":"hi"}}}
{"jsonrpc":"2.0","id":3,"method":"tool_call","params":{"name":"read_file","args":{"path":"README.md"}}}
exit
EOF
) | sed -n '1,80p'

echo "[done]"


