#!/bin/bash

set -euxo pipefail

export FLOWTRACERX_ROOT="$PWD"
export WORKSPACE="$PWD/../workspace"
cd "$WORKSPACE/Cactus"

export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

time ./simfactory/bin/sim \
    --machine="actions-$ACCELERATOR-$REAL_PRECISION" \
    create-run FlowTracerXTests --cores 1 --num-threads 2 \
    --testsuite --select-tests=FlowTracerX

TEST_OUTPUT_DIR="$(./simfactory/bin/sim \
    --machine="actions-$ACCELERATOR-$REAL_PRECISION" \
    get-output-dir FlowTracerXTests)/TEST/sim"

if test -n "${GITHUB_ENV:-}"; then
    echo "TEST_OUTPUT_DIR=$TEST_OUTPUT_DIR" >>"$GITHUB_ENV"
fi

cat "$TEST_OUTPUT_DIR/summary.log"
# A zero-test run also reports zero failures, so require every registered
# FlowTracerX test to run and pass.
grep -q '^    Total available tests    -> 9$' \
    "$TEST_OUTPUT_DIR/summary.log"
grep -q '^    Number of tests passed   -> 9$' \
    "$TEST_OUTPUT_DIR/summary.log"
grep -q '^    Number failed            -> 0$' \
    "$TEST_OUTPUT_DIR/summary.log"

# The particle-output test is also an I/O gate: it must create a non-empty
# openPMD/ADIOS2 BP5 series, not merely finish without an error.
PARTICLE_OUTPUT="$(find "$TEST_OUTPUT_DIR" -type d \
    -name 'flowtracerx_particles.it*.bp5' -print -quit)"
test -n "$PARTICLE_OUTPUT"
test -n "$(find "$PARTICLE_OUTPUT" -type f -print -quit)"
