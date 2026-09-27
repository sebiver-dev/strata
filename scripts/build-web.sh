#!/usr/bin/env bash
# Builds the browser version into web/pkg. Serve the web/ folder with any static
# file server, for example: python3 -m http.server -d web 8080
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-release}"
flag=""; [ "$profile" = "release" ] && flag="--release"
cargo build --lib --target wasm32-unknown-unknown $flag
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/wasm32-unknown-unknown/$profile/strata.wasm"
echo "Built web/pkg. Serve web/ and open it in a WebGPU browser."
