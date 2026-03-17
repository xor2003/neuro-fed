#!/usr/bin/env bash
set -euo pipefail

rm -f neurofed.db detail.log

cargo run --bin learning_benchmark -- \
  --study-paths study/minimal_pc/data/minimal_pc_sum.jsonl \
  --minimal-pc
