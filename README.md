# FlowTracerX

[![CI](https://github.com/jaykalinani/FlowTracerX/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jaykalinani/FlowTracerX/actions/workflows/ci.yml)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jaykalinani/FlowTracerX)

FlowTracerX is a GPU-accelerated passive fluid-particle tracer for
[AsterX](https://github.com/jaykalinani/AsterX).

## Thorns

- `FlowTracerX` – Particle evolution and output.
- `TestFlowTracerX` – Analytic and lifecycle tests.

## Quick Start

Add the following entries to the Cactus ThornList:

```text
FlowTracerX/FlowTracerX
FlowTracerX/TestFlowTracerX
```

Run the [Kelvin–Helmholtz instability with AMR](FlowTracerX/par/KHI_AMR.par)
example.
