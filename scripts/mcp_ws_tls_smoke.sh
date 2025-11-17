#!/usr/bin/env zsh
set -euo pipefail

# Config
ADDR=${LEXON_MCP_WS_ADDR:-127.0.0.1:9443}
BEARER=${LEXON_MCP_AUTH_BEARER:-secret}
CERT_DIR=${CERT_DIR:-tmp/certs}
CERT=${CERT_DIR}/cert.pem
KEY=${CERT_DIR}/key.pem

mkdir -p "$CERT_DIR"

if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
  echo "[gen] self-signed cert in ${CERT_DIR}"
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" -days 1 -subj "/CN=localhost" >/dev/null 2>&1
fi

echo "[start] WS TLS server at $ADDR (bearer: $BEARER)"
LEXON_MCP_WS_ADDR="$ADDR" \
LEXON_MCP_HEARTBEAT_MS=500 \
LEXON_MCP_SERVER_ONESHOT=1 \
LEXON_MCP_TLS_CERT="$CERT" \
LEXON_MCP_TLS_KEY="$KEY" \
LEXON_MCP_AUTH_BEARER="$BEARER" \
./target/debug/lexc --mcp-ws &>/tmp/lexon_ws_tls.log &
srv_pid=$!
sleep 0.4

echo "[client] websocat handshake (requires websocat installed)"
if ! command -v websocat >/dev/null 2>&1; then
  echo "websocat not found. Install with: brew install websocat (macOS) or apt-get install websocat (Linux)" >&2
  echo "[note] You can also test handshake with openssl s_client manually."
else
  # Trust our self-signed cert
  WEBSOCAT_SSL_VERIFY=0 websocat -H="Authorization: Bearer ${BEARER}" "wss://${ADDR}" -q --ping-interval 0.3 --max-messages 3 || true
fi

echo "[stop] Killing WS server ($srv_pid)"
kill $srv_pid 2>/dev/null || true
wait $srv_pid 2>/dev/null || true

echo "[done]"


