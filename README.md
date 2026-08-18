# FlowTracerX

**FlowTracerX** is a GPU-accelerated passive fluid-particle tracer for
[AsterX](https://github.com/jaykalinani/AsterX) and other HydroBaseX/ADMBaseX
systems running on [CarpetX](https://github.com/EinsteinToolkit/CarpetX).
Particles are stored and distributed with AMReX and follow
`dx^i/dt = alpha v^i - beta^i`.

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
