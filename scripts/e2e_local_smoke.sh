#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_FILE="${LOG_FILE:-/tmp/neurofed_e2e.log}"
BIN="${BIN:-target/release/neuro-fed-node}"

if [[ ! -x "$BIN" ]]; then
  echo "Building release binary with metal feature..."
  cargo build --release --features metal
fi

cleanup() {
  if [[ -n "${NODE_PID:-}" ]]; then
    kill "$NODE_PID" >/dev/null 2>&1 || true
    wait "$NODE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

env GPU_BACKEND=metal RUST_LOG=info "$BIN" >"$LOG_FILE" 2>&1 &
NODE_PID=$!

for _ in {1..60}; do
  if curl -sS http://127.0.0.1:8080/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

check_answer() {
  local question="$1"
  local expected="$2"
  local response
  response="$(curl -sS http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"neuro-fed-node\",\"messages\":[{\"role\":\"user\",\"content\":\"$question\"}],\"max_tokens\":220,\"temperature\":0.1}")"
  echo "Q: $question"
  echo "$response"
  echo "$response" | grep -qi "$expected"
}

check_answer "multiply 17 * 23" "391"
check_answer "reverse neurofed" "deforuen"
check_answer "Give a short explanation of predictive coding and one practical coding use case." "prediction error"

echo "E2E local smoke passed."
