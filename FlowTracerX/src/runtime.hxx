/**
 * \file runtime.hxx
 * \brief Process-local state shared by FlowTracerX schedule routines.
 */
#ifndef FLOWTRACERX_RUNTIME_HXX
#define FLOWTRACERX_RUNTIME_HXX

#include "carpetx_adapter.hxx"
#include "particle_container.hxx"

#include <memory>
#include <vector>

namespace FlowTracerX {

struct TracerSetState {
  long injections = 0;
};

struct Runtime {
  std::vector<std::unique_ptr<ParticleContainer>> containers;
  std::vector<TracerSetState> tracer_sets;
  amrex::Long next_id = 1;

  FieldRef gf_vtilde[3];
  FieldRef gf_alp;
  FieldRef gf_mask;
  std::vector<FieldRef> sample_gfs;

  bool initialized = false;
  bool recovered = false;
  int recovered_iteration = -1;
};

Runtime &runtime();
void resolve_runtime_fields();
TransportFields
transport_fields(const CarpetX::GHExt::PatchData::LevelData &level);
const char *configured_ode_method();
TimeIntegrator configured_time_integrator();

} // namespace FlowTracerX

#endif
