#!/bin/bash

set -euxo pipefail

export FLOWTRACERX_ROOT="$PWD"
export WORKSPACE="$PWD/../workspace"
cd "$WORKSPACE/Cactus"

export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

SIMULATION_NAME=FlowTracerXKHISmoke
PARAMETER_FILE="$FLOWTRACERX_ROOT/TestFlowTracerX/test/khi-amr-million-particles-smoke.par"

time ./simfactory/bin/sim \
    --machine="actions-$ACCELERATOR-$REAL_PRECISION" \
    create-run "$SIMULATION_NAME" --cores 1 --num-threads 2 \
    --parfile "$PARAMETER_FILE"

KHI_SMOKE_OUTPUT_DIR="$(./simfactory/bin/sim \
    --machine="actions-$ACCELERATOR-$REAL_PRECISION" \
    get-output-dir "$SIMULATION_NAME")"

if test -n "${GITHUB_ENV:-}"; then
    echo "KHI_SMOKE_OUTPUT_DIR=$KHI_SMOKE_OUTPUT_DIR" >>"$GITHUB_ENV"
fi

STDOUT_FILE="$(find "$KHI_SMOKE_OUTPUT_DIR" -type f \
    -name stdout.txt -print -quit)"
test -n "$STDOUT_FILE"
grep -q 'Injected 400000 particles into tracer set 0 (lower_shear_layer)' \
    "$STDOUT_FILE"
grep -q 'Injected 400000 particles into tracer set 1 (upper_shear_layer)' \
    "$STDOUT_FILE"
grep -q 'Injected 200000 particles into tracer set 2 (uniform_domain)' \
    "$STDOUT_FILE"
grep -q 'INFO (CarpetX):   level 1:' "$STDOUT_FILE"
