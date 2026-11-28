#!/usr/bin/env bash
set -euo pipefail
root=${MODEL_ROOT:-/content/model}
model=${MODEL_BIN:-$root/bin/model}
all=${MODEL_ALL_LABELS:-0}
batch=${MODEL_BATCH:-auto}
accum=${MODEL_ACCUM:-auto}
retain=${MODEL_RETAIN:-8}
loops_min=${MODEL_LOOPS_MIN:-4}
loops_max=${MODEL_LOOPS_MAX:-8}
w_abs=${MODEL_W_ABSOLUTE:-0}
nogeom_drop=${MODEL_NOGEOM_DROP:-0.15}
shards=$root/shards
runs=$root/runs

usage() {
    cat >&2 <<'EOF'
Usage: train.sh pre STEPS          pre-training on unlabelled CT, validated on the held-out unlabelled boxes
       train.sh mid STEPS          mid-training from the last pre-training checkpoint on every training source
                                   (MODEL_SHARES: instances,points,fibres,unlabelled,kaggle)
       train.sh eval STAGE         held-out measures of STAGE's last checkpoint on every held-out record
Runs go to MODEL_ROOT/runs/STAGE (train.jsonl, val.jsonl, gpu.csv, checkpoints); MODEL_BIN names the binary.
MODEL_ALL_LABELS=1 also trains on the held-out records and the evaluation lines and fibres (off by default).
EOF
    exit 2
}

link_set() {
    local name=$1 dir
    shift
    rm -rf "${shards:?}/sets/$name"
    mkdir -p "$shards/sets/$name"
    for dir in "$@"; do
        for f in "$shards/$dir"/*.shard; do
            [ -e "$f" ] && ln -s "$f" "$shards/sets/$name/"
        done
    done
}

last_checkpoint() {
    find "$runs/$1" -name 'step-*.ckpt' | sort -t- -k2 -n | tail -1
}

[ $# -ge 2 ] || usage
common=(--batch "$batch" --accum "$accum" --retain "$retain" --loops-min "$loops_min" --loops-max "$loops_max"
    --w-absolute "$w_abs" --nogeom-drop "$nogeom_drop" --all-labels "$all")
case $1 in
    pre)
        "$model" train --stage pre --shards "$shards/pre" --val "$shards/pre" --out "$runs/pre" --steps "$2" --seed 1 \
            "${common[@]}"
        ;;
    mid)
        if [ "$all" = 1 ]; then
            link_set mid mid pre eval kaggle
        else
            link_set mid mid pre kaggle
        fi
        link_set val mid pre eval kaggle
        "$model" train --stage mid --shards "$shards/sets/mid" --val "$shards/sets/val" \
            --resume "$(last_checkpoint pre)" --out "$runs/mid" --steps "$2" \
            --shares "${MODEL_SHARES:-0.35,0.25,0.1,0.15,0.15}" "${common[@]}"
        ;;
    eval)
        link_set val mid pre eval kaggle
        "$model" eval --ckpt "$(last_checkpoint "$2")" --shards "$shards/sets/val" --out "$runs/$2/eval.json"
        ;;
    *) usage ;;
esac
