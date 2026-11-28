#!/usr/bin/env bash
set -uo pipefail
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE} OMP_DYNAMIC=${OMP_DYNAMIC:-TRUE}
root=/content/unroll
cd "$root" || exit 1
mkdir -p graphs units logs tmp
setting() {
    python3 -c 'import json,sys;print(json.load(open("code/scan.json"))[sys.argv[1]])' "$1"
}
scroll=$(setting scroll)
scan=$(setting scan)
curl=$(setting curl)
voxel=$(setting voxel_um)
read -r -a extra <<<"$(cat code/trace_args.txt 2>/dev/null)"
anchor=()
if [ -f code/anchor ]; then
    anchor=(--anchor graphs)
fi
for name in "$@"; do
    if [ -s "graphs/$name/surfels.bin" ] && [ -s "units/$name/record.json" ]; then
        echo "SKIP $name" >>progress.txt
        continue
    fi
    box=$(python3 -c 'import json,sys;u={x["name"]:x for x in json.load(open("code/units_plan.json"))}[sys.argv[1]];print("%d:%d,%d:%d,%d:%d" % (u["z0"], u["z1"], u["y0"], u["y1"], u["x0"], u["x1"]))' "$name")
    echo "START $name $box $(date +%s)" >>progress.txt
    if ! python3 code/fetch.py unit "$name" "tmp/$name.npy" --work code >"logs/$name.fetch.log" 2>&1; then
        echo "FAIL fetch $name" >>progress.txt
        rm -f "tmp/$name.npy"
        continue
    fi
    if ./trace --surface "tmp/$name.npy" --axis code/axis.json --box "$box" --curl "$curl" --stride 8 \
        --voxel "$voxel" --min-nodes 200 --sample "$scroll" --scan "$scan" ${anchor[@]+"${anchor[@]}"} \
        ${extra[@]+"${extra[@]}"} --save-graph "graphs/$name" --skip-emit --out "units/$name" \
        >"logs/$name.log" 2>&1; then
        echo "DONE $name $(date +%s)" >>progress.txt
    else
        echo "FAIL trace $name" >>progress.txt
    fi
    rm -f "tmp/$name.npy"
done
echo "ALLDONE $(date +%s)" >>progress.txt
