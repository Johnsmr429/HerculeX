#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
case "${1:-}" in
    "" | -h | --help)
        printf '%s\n' 'Usage: ./run.sh setup [native|lint|spots]' \
            '       ./run.sh unwrap [--queue] [--from STAGE] [--rotate180] SCROLL[:SCAN] ...' \
            '       ./run.sh unroll SCROLL SCAN STAGE [ARGS]' \
            '       ./run.sh level SCROLL[:SCAN] ...' '       ./run.sh MODULE [ARGS]' \
            '       ./run.sh list' '       ./run.sh board --help' \
            '       ./run.sh check [cpu|all|lint]' '       ./run.sh paths [data|library|outputs]' \
            '       ./run.sh test-source tests/NAME_e2e.c'
        exit 0
        ;;
esac
export HERCULEX_HOME=$PWD
if [ "$1" = setup ] && [ "${2:-}" = spots ]; then
    if [ -x runtime/deps/duckdb ] || command -v duckdb >/dev/null; then
        exit 0
    fi
    case "$(uname -s):$(uname -m)" in
        Darwin:*)
            target=osx-universal
            digest=9cbe0962d2aa87501092010a3f43b224d2dfce9a4148b2eb1f5a9086c6e01be9
            ;;
        Linux:x86_64)
            target=linux-amd64
            digest=c1d6db2294895c97849bee574b21eb462528857ccf8a23617dd3e0a05dd4b770
            ;;
        Linux:aarch64)
            target=linux-arm64
            digest=32fea57520b1ae6caa291717d5f5d4d478fb6f8ed2d2c7054096a038fc6be833
            ;;
        *)
            printf '%s\n' 'Install DuckDB and set HERCULEX_DUCKDB to its executable.' >&2
            exit 1
            ;;
    esac
    mkdir -p runtime/deps
    work=$(mktemp -d runtime/deps/duckdb.XXXXXXXX)
    trap 'rm -rf "$work"' EXIT
    url=https://github.com/duckdb/duckdb/releases/download/v1.5.4
    curl -fsSL --retry 2 "$url/duckdb_cli-$target.gz" -o "$work/duckdb.gz"
    checksum=(sha256sum)
    if ! command -v sha256sum >/dev/null; then
        checksum=(shasum -a 256)
    fi
    printf '%s  %s\n' "$digest" "$work/duckdb.gz" | "${checksum[@]}" -c -
    gzip -dc "$work/duckdb.gz" >"$work/duckdb"
    if [ "$target" = osx-universal ]; then
        lipo "$work/duckdb" -thin "$(uname -m)" -output "$work/local"
        mv -f "$work/local" "$work/duckdb"
    fi
    chmod +x "$work/duckdb"
    mv -f "$work/duckdb" runtime/deps/duckdb
    exit 0
fi
if [ "$1" = setup ] && [ "${2:-}" = lint ]; then
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64) platform=macosx_11_0_arm64 ;;
        Linux:x86_64) platform=x86_64 ;;
        Linux:aarch64) platform=aarch64 ;;
        *)
            printf '%s\n' 'Put clang-format 23.1.5, clang-tidy 23.1.4, shellcheck 0.11.0 and shfmt 4.2.0 on PATH.' >&2
            exit 0
            ;;
    esac
    files=https://files.pythonhosted.org/packages
    wheels=(
        "clang-format macosx_11_0_arm64 0a/a9/3a17b3eac861ae4d67dd1bed59672028ac66007da056691553ab6cfa1be5/clang_format-23.1.5-py2.py3-none-macosx_11_0_arm64.whl 3fb740335d939dd02252cd5075050fc8858466b16cf9c1efa2f52da3baa816a7 clang_format/data/bin/clang-format"
        "clang-format aarch64 24/6c/ecb0739fb3f1798098f4b63c0e22aa0fd5a1d639f3ae4235f8155935a1ef/clang_format-23.1.5-py2.py3-none-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl e8037e12eae19a9779e01ca6719b8d7f6a939ade8869b061912551c9748a27ad clang_format/data/bin/clang-format"
        "clang-format x86_64 0e/cc/06adcfb7dfaea4d2c1929202021d4afcdc26915dc75fc0c72d63855da553/clang_format-23.1.5-py2.py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl d97ce49911ba8c89ef3f3c0735a9c7818e4780ab4812c67d1c5fcd829d16ae52 clang_format/data/bin/clang-format"
        "clang-tidy macosx_11_0_arm64 0c/59/637ce45a6cfa15ddb926e1feeee8077b9ebeb2956548375f24e16d51a908/clang_tidy-23.1.4-py2.py3-none-macosx_11_0_arm64.whl 08e977575e9f800c1c806f2001a3cdfd01323491aec05e86dd24be9a5c2585f4 clang_tidy/data/bin/clang-tidy"
        "clang-tidy aarch64 7d/95/ba9463f94319dddd0c3a9d3f65d432ddf71fe9b203dcd7ea63bcff8999e2/clang_tidy-23.1.4-py2.py3-none-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl 87596cd77dedfdea23ced1d13e57e869dcaee99745bfe942796196da3c197671 clang_tidy/data/bin/clang-tidy"
        "clang-tidy x86_64 80/85/9b483f300124a9245ec6da8975612f1841e26da13b27ad8d17f2006e13bd/clang_tidy-23.1.4-py2.py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 6bb31e4f319ecbe14acb75f23c3050d0f1d4b1d92ea1c66091bc76e4eac2c1ac clang_tidy/data/bin/clang-tidy"
        "shellcheck macosx_11_0_arm64 61/ac/2a84c37171c0cf5a10ea4b0a27d43eb0a1d29bd98b49c2c5ffe17ad24bbe/shellcheck_py-0.11.0.1-py2.py3-none-macosx_11_0_arm64.whl 6b88d0a244c82ed07e06a53e444da841f69330ca59ae15d4a66c391655dae7a0 shellcheck_py-0.11.0.1.data/scripts/shellcheck"
        "shellcheck x86_64 96/55/250e0e3367613a5c22bd82e33b16b889287d81ab0f7dda67e6514a4cccf4/shellcheck_py-0.11.0.1-py2.py3-none-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl 1b274df81de5b000ff78db433e7328b87e52e3c38481c60f8e488c3095beef05 shellcheck_py-0.11.0.1.data/scripts/shellcheck"
        "shfmt macosx_11_0_arm64 e3/39/3b5c4061a4fa7dcfab88ca4bd17566132a82225aad78d6d5b42cab4beee7/shfmt_py-4.2.0-py2.py3-none-macosx_11_0_arm64.whl 7b7fa81b120fdccebbe60b9c3411ce1b549fa2828f2809e8fa5e231a9bac65dd shfmt_py-4.2.0.data/scripts/shfmt"
        "shfmt aarch64 53/32/9448afcca6739c64810143d4dacae6a817308aab71d083d39e91b13b31ee/shfmt_py-4.2.0-py2.py3-none-manylinux2014_aarch64.whl e615cfcfe3e184f59e3ac2574895542976995a46f7506b327c8af71ae02de84c shfmt_py-4.2.0.data/scripts/shfmt"
        "shfmt x86_64 01/8a/03246179f8bafea1c4f0b664ef5fcad1814f2cf55af6acb2f614463eca20/shfmt_py-4.2.0-py2.py3-none-manylinux2014_x86_64.whl 867b55792952d4e4aa27a1b2d578e63b745a77e11b7742d2f6258ef123a9c8f2 shfmt_py-4.2.0.data/scripts/shfmt"
    )
    checksum=(sha256sum)
    if ! command -v sha256sum >/dev/null; then
        checksum=(shasum -a 256)
    fi
    mkdir -p runtime/tools/bin
    work=$(mktemp -d runtime/tools/wheel.XXXXXXXX)
    trap 'rm -rf "$work"' EXIT
    for wheel in "${wheels[@]}"; do
        read -r tool target path digest member <<<"$wheel"
        if [ "$target" != "$platform" ] ||
            { [ -x "runtime/tools/bin/$tool" ] && { [ "$tool" != clang-tidy ] || [ -d runtime/tools/lib/clang ]; }; }; then
            continue
        fi
        curl -fsSL --retry 2 "$files/$path" -o "$work/$tool.whl"
        printf '%s  %s\n' "$digest" "$work/$tool.whl" | "${checksum[@]}" -c - >/dev/null
        unzip -p "$work/$tool.whl" "$member" >"$work/$tool"
        chmod +x "$work/$tool"
        mv -f "$work/$tool" "runtime/tools/bin/$tool"
        if [ "$tool" = clang-tidy ]; then
            unzip -q "$work/$tool.whl" 'clang_tidy/data/lib/*' -d "$work/tidy"
            rm -rf runtime/tools/lib
            mv "$work/tidy/clang_tidy/data/lib" runtime/tools/lib
        fi
    done
    for tool in clang-format clang-tidy shellcheck shfmt; do
        if [ ! -x "runtime/tools/bin/$tool" ]; then
            printf 'No pinned %s for this machine; lint uses the one on PATH.\n' "$tool" >&2
        fi
    done
    exit 0
fi
native_build() {
    local entry=${1:-herculex.c} name
    name=$(basename "$entry" .c)
    local libs=(json-c libcurl libzstd libpng libjpeg libtiff-4 libzip libcrypto sqlite3 zlib libdeflate blosc freetype2)
    local sources=(level/pipeline.c runner/check.c storage/route_database.c scans/catalog/scan_choice.c
        scans/chunks/cached_ct.c
        level/detection/detect_nodes.c level/detection/node_kernels.c
        level/level/umbilicus.c
        level/level/piece_kernels.c level/level/network_simplex.c
        level/level/patch_agglomeration.c level/level/level_solver.c
        level/level/sheet_flattening.c level/level/piece_joiner.c
        level/level/scroll_axis.c level/level/orientation.c
        viewer/sheet_tiles.c viewer/sheet_page.c viewer/refresh_viewers.c runner/review_check.c)
    local layout_sources=(level/layout/seams.c level/layout/packing.c level/layout/axis_fit.c)
    local cuda=() flags=() link=() includes=() package_args=() checksum=(sha256sum) digest pkg extra=()
    local cc=${CC:-cc} nvcc=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
    nvcc=${nvcc:-nvcc}
    local common=(-std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
        -I. -DHX_UNITY)
    if [ "$(uname)" = Darwin ]; then
        sources+=(level/detection/nodes_apple/host.m)
        if [ -f synthetic/three_d/render_apple/kernel_gpu.m ]; then
            sources+=(synthetic/three_d/render_apple/kernel_gpu.m)
            sources+=(synthetic/injection/render_apple/inject_gpu.m)
        fi
        if [ -f synthetic/two_d/model_apple/inference.m ]; then
            sources+=(synthetic/two_d/model_apple/inference.m)
        fi
        common+=(-fobjc-arc -fblocks -Ilevel/level/threads_apple)
        flags=(-x objective-c -fobjc-arc '-DLANE_READER="ink/model/reader_apple/reader.m"')
        link=(-framework Foundation -framework Metal -framework CoreML -framework Accelerate
            -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
            '-Wl,-dead_strip,-x')
    else
        sources+=(level/detection/nodes_cuda/host.c)
        cuda+=(ink/model/reader_cuda/reader.cu)
        libs+=(lapack blas)
        common+=(-Ilevel/level/threads_cuda -fopenmp -ffunction-sections -fdata-sections)
        if [ -f synthetic/two_d/model_cuda/inference.cu ]; then
            cuda+=(synthetic/two_d/model_cuda/inference.cu)
            common+=(-DHX_SLICES_CUDA)
        fi
        flags=('-DLANE_READER="ink/model/reader_cuda/reader.c"')
        link=(-L"${CUDA_HOME:-/usr/local/cuda}/lib64" -L"${CUDA_HOME:-/usr/local/cuda}/lib64/stubs"
            -lcuda -lnvrtc -lcudart -lcublas -lcudnn
            -lstdc++ -lpthread -lm '-Wl,--gc-sections,-s')
    fi
    if [ "$entry" != herculex.c ]; then
        sources=()
        layout_sources=()
        if [ "$entry" != tests/reader_e2e.c ]; then
            cuda=()
            flags=()
            link=(-pthread -lm)
        fi
    else
        extra=(-x none "${sources[@]}" "${layout_sources[@]}")
    fi
    if [ "$(pkg-config --modversion json-c)" = 0.19 ]; then
        printf '%s\n' 'json-c 0.19 never frees an object that holds others; ./run.sh setup installs a fixed one.' >&2
        exit 1
    fi
    pkg=$(pkg-config --cflags "${libs[@]}")
    read -r -a includes <<<"$pkg"
    pkg=$(pkg-config --libs "${libs[@]}")
    read -r -a package_args <<<"$pkg"
    if ! command -v sha256sum >/dev/null; then
        checksum=(shasum -a 256)
    fi
    digest=$(
        set -o pipefail
        {
            "$cc" -MM "${common[@]}" "${includes[@]}" "${flags[@]}" "$entry" "${extra[@]}" |
                sed 's/^[^ ]*://; s/\\$//' | tr ' ' '\n' |
                awk 'NF && $0 !~ /^\//' | LC_ALL=C sort -u | xargs "${checksum[@]}" || exit
            "${checksum[@]}" run.sh "${cuda[@]}" || exit
            printf '%s\n' "$cc" "$nvcc" "${common[@]}" "${flags[@]}" "${link[@]}" "$pkg"
        } | "${checksum[@]}" | cut -c1-16
    )
    native=runtime/bin/$name-$digest
    if [ -x "$native" ]; then
        return
    fi
    mkdir -p runtime/bin
    (
        local work objects=() cu layout object
        work=$(mktemp -d "$native.XXXXXXXX")
        trap 'rm -rf "$work"' EXIT
        for cu in "${cuda[@]}"; do
            object="$work/$(basename "$cu").o"
            "$nvcc" -O3 -std=c++17 -Xcompiler -fPIC -c "$cu" -o "$object"
            objects+=("$object")
        done
        for layout in "${layout_sources[@]}"; do
            object="$work/$(basename "$layout").o"
            "$cc" "${common[@]}" "${includes[@]}" -ffp-contract=off -c "$layout" -o "$object"
            objects+=("$object")
        done
        "$cc" "${common[@]}" "${includes[@]}" -ffp-contract=off "${flags[@]}" \
            -c "$entry" -o "$work/main.o"
        "$cc" "${common[@]}" "${includes[@]}" "${sources[@]}" "$work/main.o" \
            "${objects[@]}" "${package_args[@]}" "${link[@]}" -o "$work/herculex"
        mv -f "$work/herculex" "$native"
    )
}
if [ "${1:-}" = build ]; then
    native_build
    echo "$native"
    exit 0
fi
native_runner() {
    if [ ! -x "${HERCULEX_NATIVE_BIN:-}" ]; then
        native_build
        export HERCULEX_NATIVE_BIN=$PWD/$native
    fi
}
if [ "$1" = test-source ]; then
    case "${2:-}" in
        tests/*_e2e.sh | tests/*_golden.sh)
            exec /bin/sh "$2"
            ;;
        tests/reader_e2e.c)
            native_build "$2"
            exec "$native"
            ;;
        tests/*_e2e.c | tests/*_golden.c)
            native_runner
            native_build "$2"
            exec "$native" "${HERCULEX_NATIVE_BIN:?native test runner required}"
            ;;
        *) exit 2 ;;
    esac
fi
run_native() {
    native_runner
    exec "$HERCULEX_NATIVE_BIN" "$@"
}
if [ "$1" = scans.chunks.fetch ]; then
    case "${2:-}" in
        usage | evict | resolve)
            verb=$2
            shift 2
            run_native "scans.chunks.$verb" "$@"
            ;;
    esac
    run_native "$@"
fi
case "$1" in
    synthetic.injection.papyrus | synthetic.injection.fragment | synthetic.injection.fragment.native | \
        synthetic.injection.fidelity | synthetic.injection.fidelity.native | \
        synthetic.three_d.gallery | synthetic.three_d.inject | synthetic.three_d.inject.*)
        if [ "$(uname)" != Darwin ]; then
            printf '%s requires Apple Metal.\n' "$1" >&2
            exit 2
        fi
        ;;
esac
if { [ "$1" = level.level.run_levels ] || [ "$1" = unwrapping.level.run_levels ]; } &&
    [ -n "${HERCULEX_LEVEL_PROGRAM:-}" ]; then
    shift
    exec "$HERCULEX_LEVEL_PROGRAM" "$@"
fi
if [ "$1" = unwrap ]; then
    shift
    exec unroll/unwrap.sh "$@"
fi
if [ "$1" = unroll ] || [ "$1" = unwrapping ]; then
    shift
    exec unroll/pipeline.sh "$@"
fi
case "$1" in
    level | status | catalog | paths | sheets | pieces | fit | curl | umbilicus | board | check | layout | sample | fetch | ink | \
        models | weights | ink.model.models | ink.model.weights | publish | serve | card | \
        runner.check | runner.lint | runner.keep_faces | \
        level.pipeline | runner.sheet_stage | level.unwrap | runner.pipeline | runner.unwrap | runner.job_board | runner.review_page | \
        storage.route_database | runner.timeline | timeline | runner.disk | disk | port | \
        scans.catalog.scan_choice | scans.catalog.scan_catalog | scans.catalog.product_inventory | \
        scans.catalog.server_inventory | \
        scans.transfer.fetch_products | scans.transfer.fetch_server | \
        scans.chunks.usage | scans.chunks.evict | scans.chunks.resolve | \
        runner.fetch_ahead | runner.memory_trace | runner.place_build | runner.native_ports | runner.speed | \
        data.catalog.catalog | calibrate.survey.survey_rim_excess | calibrate.survey.survey_row_power | \
        calibrate.survey.scroll_survey | calibrate.survey.survey_fetch | \
        calibration.survey.survey_rim_excess | calibration.survey.survey_row_power | \
        calibration.survey.scroll_survey | calibration.survey.survey_fetch | \
        calibrate.unwrap.sheet_wrap_study | calibrate.unwrap.sheet_wrap_survey | \
        calibrate.unwrap.sheet_wrap_truth | calibrate.unwrap.sheet_scoring | \
        calibrate.unwrap.mesh_reference | \
        calibrate.unwrap.raw_sheet_ruler | calibrate.unwrap.sheet_controls | \
        calibration.unwrap.sheet_wrap_study | calibration.unwrap.sheet_wrap_survey | \
        calibration.unwrap.sheet_wrap_truth | calibration.unwrap.sheet_scoring | \
        calibration.unwrap.mesh_reference | \
        calibration.unwrap.raw_sheet_ruler | calibration.unwrap.sheet_controls | \
        calibrate.bench.score | calibrate.bench.score_check | calibrate.bench.results | calibrate.bench.package | \
        calibrate.bench.truth | calibrate.bench.surface_score | calibrate.bench.winding_score | \
        calibrate.bench.ridge | calibrate.bench.qa | calibrate.bench.v1_sources | \
        calibrate.text.line_spectrum | calibrate.text.readability_judge | \
        calibration.bench.score | calibration.bench.score_check | calibration.bench.results | calibration.bench.package | \
        calibration.bench.truth | calibration.bench.surface_score | calibration.bench.winding_score | \
        calibration.bench.ridge | calibration.bench.qa | calibration.bench.v1_sources | \
        calibration.text.line_spectrum | calibration.text.readability_judge | \
        synthetic.* | \
        level.detection.detect_nodes | level.level.run_levels | level.level.umbilicus | \
        level.layout.seams | level.layout.mosaic_layout | level.level.scroll_layout | \
        unwrapping.detection.detect_nodes | unwrapping.level.run_levels | unwrapping.level.umbilicus | \
        unwrapping.layout.seams | unwrapping.layout.mosaic_layout | unwrapping.level.scroll_layout | \
        viewer.sheet_tiles | viewer.refresh_viewers | viewer.site_server | viewer.survey_status | \
        viewer.site_pages | viewer.panel_pages | viewer.overview | viewer.window_proposals | \
        viewer.mosaic_outline | viewer.atlas_build | \
        website.sheet_tiles | website.refresh_viewers | website.site_server | website.survey_status | \
        website.site_pages | website.panel_pages | website.overview | website.window_proposals | \
        website.mosaic_outline | website.atlas_build | \
        spots | spots.database | spots.catalogue | spots.pilot | spots.reflatten | spots.flatten)
        run_native "$@"
        ;;
esac
module=$1
shift
case "$module" in
    setup)
        case "${1:-native}" in
            -h | --help)
                printf '%s\n' 'Usage: ./run.sh setup [native|lint|spots]'
                exit 0
                ;;
            native) ;;
            *)
                printf '%s\n' 'Usage: ./run.sh setup [native|lint|spots]' >&2
                exit 2
                ;;
        esac
        if [ "$#" -gt 1 ]; then
            printf '%s\n' 'Usage: ./run.sh setup [native|lint|spots]' >&2
            exit 2
        fi
        system_libs=(libtiff-4 json-c libcurl libcrypto zlib sqlite3 libzstd libdeflate libzip libpng libjpeg blosc freetype2)
        if [ "$(uname)" != Darwin ]; then
            system_libs+=(lapack blas)
        fi
        if ! command -v pkg-config >/dev/null || ! pkg-config --exists "${system_libs[@]}"; then
            if [ "$(uname)" = Darwin ] && command -v brew >/dev/null; then
                brew install pkgconf libtiff json-c openssl zstd libdeflate libzip libpng jpeg-turbo c-blosc freetype
            elif [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
                apt-get update
                apt-get install -y --no-install-recommends pkg-config libtiff-dev liblapack-dev libblas-dev libjson-c-dev \
                    libcurl4-openssl-dev libssl-dev zlib1g-dev libsqlite3-dev \
                    libzstd-dev libdeflate-dev libzip-dev libpng-dev libjpeg-dev libblosc-dev libfreetype-dev
            else
                printf '%s\n' 'Install pkg-config and the development packages for TIFF, json-c,' \
                    'libcurl, OpenSSL, zlib, SQLite, zstd, libdeflate, libzip, PNG, JPEG and Blosc.' \
                    'Then repeat ./run.sh setup.' >&2
                exit 1
            fi
        fi
        if [ "$(pkg-config --modversion json-c)" = 0.19 ] && command -v brew >/dev/null; then
            brew unlink json-c
            brew install --HEAD json-c
        fi
        run_native weights fetch
        ;;
    list | runner.modules)
        module=runner.modules
        native_runner
        export HERCULEX_LIST_NATIVE
        HERCULEX_LIST_NATIVE=$("$HERCULEX_NATIVE_BIN" --list-native)
        ;;
esac
packages=()
dependencies=()
build_sources=()
dispatch=()
implementation=$module
case "$module" in
    runner.modules) native_command=yes ;;
    scans.transfer.fetch_volume | ink.reading.unit_fetch)
        native_command=ct_fetch
        implementation=ink.reading.unit_fetch
        dispatch=("$module")
        packages=(json-c libcurl libcrypto libtiff-4)
        dependencies=(scans/transfer/fetch_volume.c scans/chunks/bucket_listing.h
            storage/brick_store.c storage/data_roots.c
            sampling/touched_planes.c sampling/column_kernels.c sampling/column_sampling.c
            sampling/fibre_kernels.c level/level/sheet_flattening.c)
        build_sources=(sampling/column_sampling.c)
        ;;
    scans.transfer.fetch_fragment)
        native_command=fetch_fragment
        packages=(json-c libcurl)
        dependencies=(scans/chunks/bucket_listing.h)
        ;;
    scans.transfer.fetch_atlas)
        native_command=fetch_atlas
        packages=(json-c libcurl zlib)
        dependencies=(scans/catalog/scan_catalog.c scans/transfer/fetch_volume.c
            scans/chunks/bucket_listing.h storage/brick_store.c storage/data_roots.c)
        ;;
    logs.runs.2026-11-18-vast.scripts.fit_stats)
        native_command=stats
        packages=(json-c)
        ;;
    runner.wait_for_queue | calibrate.unwrap.sheet_wrap_report | calibration.unwrap.sheet_wrap_report)
        native_command=sqlite
        if [ "$module" = calibration.unwrap.sheet_wrap_report ]; then
            implementation=calibrate.unwrap.sheet_wrap_report
        fi
        ;;
    calibrate.unwrap.continuous_area | calibration.unwrap.continuous_area)
        native_command=area
        if [ "$module" = calibration.unwrap.continuous_area ]; then
            implementation=calibrate.unwrap.continuous_area
        fi
        packages=(libtiff-4 json-c)
        dependencies=(storage/data_roots.c sampling/column_sampling.c sampling/fibre_kernels.c
            level/level/sheet_flattening.c)
        ;;
    level.layout.tifxyz_export)
        native_command=tifxyz
        packages=(libtiff-4 json-c libcrypto)
        dependencies=(storage/data_roots.c sampling/column_sampling.c)
        ;;
    *) native_command= ;;
esac
if [ -n "$native_command" ]; then
    source=${implementation//./\/}.c
    if [ ! -f "$source" ]; then
        printf 'Command %s is unavailable in this checkout (missing %s).\n' "$module" "$source" >&2
        exit 2
    fi
fi
if [ -n "$native_command" ]; then
    for dependency in "${dependencies[@]}"; do
        if [ ! -f "$dependency" ]; then
            printf 'Command %s is unavailable in this checkout (missing %s).\n' "$module" "$dependency" >&2
            exit 2
        fi
    done
    checksum=(sha256sum)
    if ! command -v sha256sum >/dev/null; then
        checksum=(shasum -a 256)
    fi
    set -o pipefail
    digest=$(cat "$source" "${dependencies[@]}" run.sh | "${checksum[@]}")
    native="runtime/commands/$implementation-$(uname -m)-${digest%% *}"
    if [ ! -x "$native" ]; then
        flags=()
        if ((${#packages[@]})); then
            options=$(pkg-config --cflags --libs "${packages[@]}")
            read -r -a flags <<<"$options"
        elif [ "$native_command" = sqlite ]; then
            flags=(-lsqlite3)
        fi
        if [ "$native_command" = fetch_fragment ]; then
            flags+=(-pthread)
        fi
        if [ "$native_command" = fetch_atlas ]; then
            flags+=(-DHX_FETCH_ATLAS_CLI -pthread -lm)
        fi
        if [ "$native_command" = area ]; then
            flags+=(-ffp-contract=off -pthread -lm)
        fi
        if [ "$native_command" = ct_fetch ]; then
            flags+=(-DHX_COLUMN_NATIVE -DHX_MESH_BRIDGE -ffp-contract=off -pthread -lm)
            if [ "$(uname)" = Darwin ]; then
                flags+=(-fblocks)
            else
                flags+=(-fopenmp)
            fi
        fi
        mkdir -p runtime/commands
        part=$(mktemp "$native.XXXXXXXX")
        trap 'rm -f "$part"' EXIT
        "${CC:-cc}" -std=gnu11 -O2 -Wall -Wextra "$source" "${build_sources[@]}" \
            "${flags[@]}" -o "$part"
        chmod +x "$part"
        mv -f "$part" "$native"
    fi
    exec "$native" "${dispatch[@]}" "$@"
fi
printf 'Unknown command %s. Run ./run.sh list.\n' "$module" >&2
exit 2
