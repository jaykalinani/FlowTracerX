#!/bin/bash

set -euxo pipefail

export FLOWTRACERX_ROOT="$PWD"
export WORKSPACE="$PWD/../workspace"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

wget https://raw.githubusercontent.com/gridaphobe/CRL/master/GetComponents
chmod a+x GetComponents
./GetComponents --no-parallel --shallow \
    "$FLOWTRACERX_ROOT/scripts/flowtracerx.th"

cd Cactus
mkdir -p arrangements/FlowTracerX
pushd arrangements/FlowTracerX
for thorn in FlowTracerX TestFlowTracerX; do
    test -f "$FLOWTRACERX_ROOT/$thorn/interface.ccl"
    ln -s "$FLOWTRACERX_ROOT/$thorn" .
    test -f "$thorn/interface.ccl"
done
popd
