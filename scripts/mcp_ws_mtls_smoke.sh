#!/usr/bin/env zsh
set -euo pipefail

ADDR=${LEXON_MCP_WS_ADDR:-127.0.0.1:9444}
CERT_DIR=${CERT_DIR:-tmp/mtls}
mkdir -p "$CERT_DIR"

# Generate CA, server cert, and client cert
CA_KEY="$CERT_DIR/ca.key.pem"
CA_CERT="$CERT_DIR/ca.cert.pem"
SRV_KEY="$CERT_DIR/server.key.pem"
SRV_CSR="$CERT_DIR/server.csr.pem"
SRV_CERT="$CERT_DIR/server.cert.pem"
CLI_KEY="$CERT_DIR/client.key.pem"
CLI_CSR="$CERT_DIR/client.csr.pem"
CLI_CERT="$CERT_DIR/client.cert.pem"

if [[ ! -f "$CA_CERT" ]]; then
  echo "[gen] CA"
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$CA_KEY" -out "$CA_CERT" -days 1 -subj "/CN=lexon-ca" >/dev/null 2>&1
fi
if [[ ! -f "$SRV_CERT" ]]; then
  echo "[gen] server cert"
  openssl req -newkey rsa:2048 -nodes -keyout "$SRV_KEY" -out "$SRV_CSR" -subj "/CN=localhost" >/dev/null 2>&1
  openssl x509 -req -in "$SRV_CSR" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial -out "$SRV_CERT" -days 1 >/dev/null 2>&1
fi
if [[ ! -f "$CLI_CERT" ]]; then
  echo "[gen] client cert"
  openssl req -newkey rsa:2048 -nodes -keyout "$CLI_KEY" -out "$CLI_CSR" -subj "/CN=client" >/dev/null 2>&1
  openssl x509 -req -in "$CLI_CSR" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial -out "$CLI_CERT" -days 1 >/dev/null 2>&1
fi

echo "[start] WS mTLS server at $ADDR"
LEXON_MCP_WS_ADDR="$ADDR" \
LEXON_MCP_HEARTBEAT_MS=0 \
LEXON_MCP_SERVER_ONESHOT=1 \
LEXON_MCP_TLS_CERT="$SRV_CERT" \
LEXON_MCP_TLS_KEY="$SRV_KEY" \
LEXON_MCP_TLS_CLIENT_CA="$CA_CERT" \
./target/debug/lexc --mcp-ws &>/tmp/lexon_ws_mtls.log &
srv=$!
sleep 0.4

echo "[client] openssl s_client with client cert (with timeout)"
# Prefer GNU timeout if available; fallback to BSD timeout if present; otherwise manual kill
if command -v gtimeout >/dev/null 2>&1; then
  TMO=gtimeout
elif command -v timeout >/dev/null 2>&1; then
  TMO=timeout
else
  TMO=
fi
if [[ -n "${TMO}" ]]; then
  printf "GET / HTTP/1.1\r\nHost: ${ADDR}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==\r\n\r\n" \
   | ${TMO} 3 openssl s_client -connect "${ADDR}" -servername 127.0.0.1 -quiet -cert "$CLI_CERT" -key "$CLI_KEY" -CAfile "$CA_CERT" 2>/dev/null \
   | sed -n '1,20p' || true
else
  # Fallback: run in background and kill after 3s
  ( printf "GET / HTTP/1.1\r\nHost: ${ADDR}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==\r\n\r\n" \
   | openssl s_client -connect "${ADDR}" -servername 127.0.0.1 -quiet -cert "$CLI_CERT" -key "$CLI_KEY" -CAfile "$CA_CERT" 2>/dev/null \
   | sed -n '1,20p' ) & pid=$!
  sleep 3
  kill $pid 2>/dev/null || true
fi

echo "[stop] kill server"
kill $srv 2>/dev/null || true
wait $srv 2>/dev/null || true

echo "[done]"


