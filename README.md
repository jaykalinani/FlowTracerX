# FlowTracerX

FlowTracerX is a GPU-native passive fluid-particle tracer for AsterX and other
HydroBaseX/ADMBaseX evolution systems on CarpetX. CarpetX and the in-house
ODESolvers/Subcycling thorns own scheduling, Runge--Kutta stages, AMR, and
level time alignment. AMReX supplies pure-SoA particle storage, GPU tile
kernels, ownership migration, and restart; AMReX time subcycling is not used.

The one-million-tracer single-GPU target has been exercised successfully.
Weak scaling toward thousand-GPU, billion-particle jobs remains an unmeasured
architecture target.

## Thorn layout

- `FlowTracerX/`: production thorn.
- `TestFlowTracerX/`: analytic GPU and lifecycle tests.

## Build

Add these local thorns to the Cactus ThornList:

```text
FlowTracerX/FlowTracerX
FlowTracerX/TestFlowTracerX   # tests only
```

The production thorn requires CarpetX, AMReX, Loop, HydroBaseX, ADMBaseX,
ODESolvers, MPI, and IOUtil. openPMD diagnostics are compiled when the
`openPMD_api` capability is available.

## Quick start

[`FlowTracerX/par/spherical_shock_unigrid_tracers.par`](FlowTracerX/par/spherical_shock_unigrid_tracers.par)
is the full 160^3 AsterX/RePrimAnd example with 1000 tracers in the dense core.
[`FlowTracerX/par/KHI_AMR.par`](FlowTracerX/par/KHI_AMR.par)
is the four-level, in-house-subcycled Kelvin--Helmholtz example. It seeds 200
particles across the lower shear layer and 800 across the upper layer.
[`FlowTracerX/par/KHI_AMR_million_particles.par`](FlowTracerX/par/KHI_AMR_million_particles.par)
is the single-GPU scale case with 400,000 tracers around each shear layer and
200,000 distributed uniformly over the domain. These
examples use GPU interpolation, AsterMasks deletion, absorbing physical
boundaries, and synchronized BP5 diagnostics. The spherical case also uses
AMReX particle sidecars driven by ordinary `IO::checkpoint_*` settings. The
KHI visualization case disables checkpoints because this audited CarpetX
revision asserts while checkpointing its changing AMR band mesh; dedicated
FlowTracerX checkpoint/recovery tests remain enabled and passing.

A minimal domain-wide random tracer set is:

```text
FlowTracerX::active = yes
FlowTracerX::num_tracer_sets = 1
FlowTracerX::tracer_name[0] = "fluid_markers"
FlowTracerX::seed_type[0] = "random_box"
FlowTracerX::num_particles[0] = 1000000
```

The transport law is `dx^i/dt = vtilde^i = alpha v^i - beta^i`. HydroBaseX
velocity is cell centered; ADMBaseX lapse and shift are averaged from vertices
to cells in a CarpetX device loop. Particle interpolation, masks, and boundary
handling execute in AMReX GPU kernels—there is no host or global Cactus
particle interpolation path.
