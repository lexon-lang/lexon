#!/usr/bin/env zsh
set -euo pipefail

export LEXON_MCP_HEARTBEAT_MS=${LEXON_MCP_HEARTBEAT_MS:-0}
export LEXON_MCP_RATE_PER_MIN=${LEXON_MCP_RATE_PER_MIN:-5}
export LEXON_MCP_AUTH_JWT_HS256=${LEXON_MCP_AUTH_JWT_HS256:-secret}

# Build a short-lived HS256 JWT (sub: demo, exp: now+60)
JWT=$(python3 - << 'PY'
import base64, json, hmac, hashlib, time, os, sys
secret=os.environ.get("LEXON_MCP_AUTH_JWT_HS256","secret").encode()
def b64u(x): return base64.urlsafe_b64encode(x).rstrip(b'=')
header=b64u(json.dumps({"alg":"HS256","typ":"JWT"}).encode())
payload=b64u(json.dumps({"sub":"demo","exp":int(time.time())+60}).encode())
hp=header+b"."+payload
sig=hmac.new(secret, hp, hashlib.sha256).digest()
jwt=hp+b"."+b64u(sig)
print(jwt.decode())
PY
)

echo "[start] stdio server (JWT auth)"
(
  ./target/debug/lexc --mcp-stdio << EOF
{"jsonrpc":"2.0","id":1,"method":"list_tools"}
{"jsonrpc":"2.0","id":2,"method":"auth","params":{"jwt":"$JWT"}}
{"jsonrpc":"2.0","id":3,"method":"list_tools"}
exit
EOF
) | sed -n '1,80p'

echo "[done]"


