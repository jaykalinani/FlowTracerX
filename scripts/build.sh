#!/bin/bash

set -euxo pipefail

export FLOWTRACERX_ROOT="$PWD"
export WORKSPACE="$PWD/../workspace"
cd "$WORKSPACE/Cactus"

FLOWTRACERX_SCRIPTS="$FLOWTRACERX_ROOT/scripts"
cp "$FLOWTRACERX_SCRIPTS/actions-$ACCELERATOR-$REAL_PRECISION.cfg" \
    simfactory/mdb/optionlists/
cp "$FLOWTRACERX_SCRIPTS/actions-$ACCELERATOR-$REAL_PRECISION.ini" \
    simfactory/mdb/machines/
cp "$FLOWTRACERX_SCRIPTS/actions-real64.run" \
    simfactory/mdb/runscripts/
cp "$FLOWTRACERX_SCRIPTS/actions-real64.sub" \
    simfactory/mdb/submitscripts/
cp "$FLOWTRACERX_SCRIPTS/defs.local.ini" simfactory/etc/
cp "$FLOWTRACERX_SCRIPTS/flowtracerx.th" .

# Keep the analytic test thorn out of production executables.
printf '\nFlowTracerX/TestFlowTracerX\n' >>flowtracerx.th

if command -v ccache >/dev/null 2>&1; then
    export CCACHE_DIR="${CCACHE_DIR:-$FLOWTRACERX_ROOT/.ccache}"
    ccache --max-size=2G
    ccache --zero-stats || true
    sed -i -e 's/^CC = /CC = ccache /' -e 's/^CXX = /CXX = ccache /' \
        "simfactory/mdb/optionlists/actions-$ACCELERATOR-$REAL_PRECISION.cfg"
fi

time ./simfactory/bin/sim \
    --machine="actions-$ACCELERATOR-$REAL_PRECISION" \
    build -j "$(nproc)" sim 2>&1 | tee build.log

test -x exe/cactus_sim
command -v ccache >/dev/null 2>&1 && ccache --show-stats || true
