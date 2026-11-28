#!/usr/bin/env bash
set -u
root=${MODEL_ROOT:-/content/model}
jobs=$root/jobs
code=$root/code
bin=${MODEL_BIN:-$root/bin/model}
raw=$root/raw/kaggle-surface
shard=$root/shards/kaggle/kaggle.shard
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$jobs/chain.log"; }
size() { stat -c %s "$1" 2>/dev/null || echo 0; }

mkdir -p "$jobs"
if [ "$(size "$shard")" -lt 1000000 ]; then
    if [ -f "$jobs/kaggle.log" ]; then
        log "waiting for the Kaggle download (job kaggle)"
        until [ -f "$jobs/kaggle.rc" ]; do sleep 60; done
    fi
    mkdir -p "$root/shards/kaggle"
    if [ "$(find "$raw/train_images" -name '*.tif' 2>/dev/null | wc -l)" -ge 700 ]; then
        log "converting Kaggle volumes"
        rm -f "$shard.part"
        "$root/bin/prep" kaggle "$raw" "$shard.part" --csv "$raw/train.csv" --holdout-every 10 --delete-raw 1 \
            >>"$jobs/chain.log" 2>&1
        [ "$(size "$shard.part")" -ge 1000000 ] && mv "$shard.part" "$shard"
    fi
fi
shares=0.35,0.25,0.1,0.15,0.15
if [ "$(size "$shard")" -lt 1000000 ]; then
    rm -f "$shard"
    shares=0.4,0.25,0.15,0.2,0
fi
log "Kaggle shard: $(find "$root/shards/kaggle" -maxdepth 1 -type f -printf '%f %s bytes ' 2>/dev/null) shares $shares"
log "waiting for pre-training (job pre)"
until [ -f "$jobs/pre.rc" ]; do sleep 60; done
log "pre-training ended with $(cat "$jobs/pre.rc")"
steps=$(cat "$jobs/mid_steps" 2>/dev/null || echo "${MODEL_MID_STEPS:-50000}")
log "mid-training starts, $steps steps"
(cd "$code" && MODEL_BIN=$bin MODEL_SHARES=$shares bash train.sh mid "$steps") >"$jobs/mid.log" 2>&1
echo $? >"$jobs/mid.rc"
log "mid-training ended with $(cat "$jobs/mid.rc")"
(cd "$code" && MODEL_BIN=$bin bash train.sh eval mid) >"$jobs/eval.log" 2>&1
echo $? >"$jobs/eval.rc"
last=$(find "$root/runs/mid" -name 'step-*.ckpt' | sort -t- -k2 -n | tail -1)
"$bin" dump --ckpt "$last" --shards "$root/shards/sets/val" --out "$root/runs/mid/dump" --per-source 6 >"$jobs/dump.log" 2>&1
echo $? >"$jobs/dump.rc"
log "eval and dump done"
