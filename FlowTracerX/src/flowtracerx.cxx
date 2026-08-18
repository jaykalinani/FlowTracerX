/**
 * \file flowtracerx.cxx
 * \brief Cactus scheduling, inherited-field access, and public API.
 */
#include "runtime.hxx"

#include "flowtracerx_api.hxx"
#include "seeding.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_ParallelDescriptor.H>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <string>

namespace FlowTracerX {

Runtime &runtime() {
  static Runtime state;
  return state;
}

const char *configured_ode_method() {
  int parameter_type = -1;
  const void *const parameter =
      CCTK_ParameterGet("method", "ODESolvers", &parameter_type);
  if (!parameter || parameter_type != PARAMETER_KEYWORD)
    CCTK_ERROR("FlowTracerX could not read ODESolvers::method");
  return *static_cast<const char *const *>(parameter);
}

TimeIntegrator configured_time_integrator() {
  const char *const ode_method = configured_ode_method();
  return CCTK_EQUALS(ode_method, "RK4") || CCTK_EQUALS(ode_method, "IMEX42L")
             ? TimeIntegrator::rk4
             : TimeIntegrator::ssprk3;
}

void resolve_runtime_fields() {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  state.gf_vtilde[0] = resolve_field("FlowTracerX::vtilde_x");
  state.gf_vtilde[1] = resolve_field("FlowTracerX::vtilde_y");
  state.gf_vtilde[2] = resolve_field("FlowTracerX::vtilde_z");
  state.gf_alp = resolve_field("ADMBaseX::alp");
  state.gf_mask = resolve_field(destroy_mask_var, true);

  state.sample_gfs.clear();
  std::set<std::string> output_names;
  std::istringstream variables(sample_variables);
  for (std::string name; variables >> name;) {
    const FieldRef field = resolve_field(name);
    if (!output_names.insert(field.output_name).second)
      throw std::runtime_error(
          "FlowTracerX: diagnostic variables must have unique short names; \"" +
          field.output_name + "\" appears more than once");
    state.sample_gfs.push_back(field);
  }
}

TransportFields
transport_fields(const CarpetX::GHExt::PatchData::LevelData &level) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  TransportFields fields;
  for (int d = 0; d < 3; ++d) {
    fields.vtilde[d] = &field_mfab(level, state.gf_vtilde[d]);
    fields.vtilde_component[d] = state.gf_vtilde[d].component_index;
    const auto nodal = field_nodal_flags(level, state.gf_vtilde[d]);
    for (int axis = 0; axis < 3; ++axis)
      fields.vtilde_centering[d].nodal[axis] = nodal[axis];
  }

  if (destroy_if_lapse_below >= 0) {
    fields.alp = &field_mfab(level, state.gf_alp);
    fields.alp_component = state.gf_alp.component_index;
    const auto nodal = field_nodal_flags(level, state.gf_alp);
    for (int axis = 0; axis < 3; ++axis)
      fields.alp_centering.nodal[axis] = nodal[axis];
  }
  if (state.gf_mask) {
    fields.mask = &field_mfab(level, state.gf_mask);
    fields.mask_component = state.gf_mask.component_index;
    const auto nodal = field_nodal_flags(level, state.gf_mask);
    for (int axis = 0; axis < 3; ++axis)
      fields.mask_centering.nodal[axis] = nodal[axis];
  }
  fields.mask_below = CCTK_EQUALS(destroy_mask_comparison, "below");
  fields.mask_value = destroy_mask_value;
  fields.alp_floor = destroy_if_lapse_below;
  return fields;
}

extern "C" void FlowTracerX_ParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (!active)
    return;
  const char *const ode_method = configured_ode_method();
  if (!(CCTK_EQUALS(ode_method, "RK4") ||
        CCTK_EQUALS(ode_method, "SSPRK3") ||
        CCTK_EQUALS(ode_method, "IMEX42L") ||
        CCTK_EQUALS(ode_method, "IMEX32L")))
    CCTK_PARAMWARN("FlowTracerX requires ODESolvers::method to be RK4, "
                   "SSPRK3, IMEX42L, or IMEX32L");
  if (num_tracer_sets < 0 || num_tracer_sets > 32)
    CCTK_PARAMWARN(
        "FlowTracerX::num_tracer_sets must be between 0 and 32");
  if (use_subcycling) {
    const int required_ghostzones = subcycling_guard_cells + 1;
    if (cctk_nghostzones[0] < required_ghostzones ||
        cctk_nghostzones[1] < required_ghostzones ||
        cctk_nghostzones[2] < required_ghostzones)
      CCTK_PARAMWARN(
          "FlowTracerX subcycling requires at least "
          "subcycling_guard_cells + 1 allocated ghost zones in every "
          "direction for the GPU trilinear stencil");
  }

  std::set<std::string> names;
  for (int tracer_set = 0; tracer_set < num_tracer_sets; ++tracer_set) {
    if (tracer_name[tracer_set][0] != '\0' &&
        !names.insert(tracer_name[tracer_set]).second)
      CCTK_PARAMWARN("FlowTracerX tracer names must be unique");
    if (injection_every_iterations[tracer_set] > 0 &&
        injection_every_time[tracer_set] > 0)
      CCTK_PARAMWARN("A FlowTracerX tracer set cannot use both iteration "
                     "and time injection cadences");
    if (CCTK_EQUALS(seed_type[tracer_set], "random_shell") &&
        seed_inner_radius[tracer_set] >= seed_radius[tracer_set])
      CCTK_PARAMWARN("FlowTracerX random shells require "
                     "seed_inner_radius < seed_radius");
    if ((CCTK_EQUALS(seed_type[tracer_set], "random_sphere") ||
         CCTK_EQUALS(seed_type[tracer_set], "random_shell")) &&
        num_particles[tracer_set] > 0 && seed_radius[tracer_set] <= 0)
      CCTK_PARAMWARN(
          "FlowTracerX sphere and shell seeds require seed_radius > 0");

    if (CCTK_EQUALS(seed_type[tracer_set], "lattice")) {
      const auto nx = static_cast<amrex::Long>(lattice_nx[tracer_set]);
      const auto ny = static_cast<amrex::Long>(lattice_ny[tracer_set]);
      const auto nz = static_cast<amrex::Long>(lattice_nz[tracer_set]);
      const auto limit = std::numeric_limits<amrex::Long>::max();
      if (nx > limit / ny || nx * ny > limit / nz)
        CCTK_PARAMWARN("FlowTracerX lattice particle count overflows the "
                       "AMReX particle index type");
    }
  }
#ifndef HAVE_CAPABILITY_openPMD_api
  if (out_every > 0)
    CCTK_PARAMWARN("FlowTracerX openPMD output was requested, but this "
                   "configuration lacks openPMD_api");
#endif
  try {
    resolve_runtime_fields();
  } catch (const std::exception &error) {
    CCTK_PARAMWARN(error.what());
  }
}

extern "C" void FlowTracerX_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (CarpetX::ghext->num_patches() != 1)
    CCTK_VERROR("FlowTracerX currently supports single-patch Cartesian "
                "CarpetX runs; a batched GPU map is required for %d patches",
                CarpetX::ghext->num_patches());

  if (state.containers.empty()) {
    state.containers.reserve(CarpetX::ghext->num_patches());
    for (auto &patch : CarpetX::ghext->patchdata)
      state.containers.emplace_back(
          std::make_unique<ParticleContainer>(patch.amrcore.get()));
  } else {
    for (auto &container : state.containers)
      container->resize_after_regrid();
  }
  state.tracer_sets.resize(num_tracer_sets);
  state.initialized = true;
  if (verbose)
    CCTK_VINFO("Attached %zu AMReX pure-SoA particle container(s)",
               state.containers.size());
}

extern "C" void FlowTracerX_Initialise(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  auto &state = runtime();
  if (state.initialized && !state.recovered)
    inject_due_tracer_sets(cctkGH, true);
}

extern "C" void FlowTracerX_BeginStep(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  auto &state = runtime();
  if (!CarpetX::active_levels)
    CCTK_VERROR("FlowTracerX CCTK_PRESTEP ran without CarpetX active levels");

  // CarpetX's in-house subcycling driver forms the active level batch and
  // sets cctk_timefac before PRESTEP. CCTK_DELTA_TIME is therefore the local
  // level step, not AMReX time-subcycling state.
  const TimeIntegrator integrator = configured_time_integrator();
  const CCTK_REAL dt = CCTK_DELTA_TIME;
  CarpetX::active_levels->loop_serially([&](auto &level) {
    state.containers.at(level.patch)
        ->begin_step(level.level, integrator, dt);
  });
}

extern "C" void FlowTracerX_AdvanceStage(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  auto &state = runtime();
  if (!CarpetX::active_levels)
    CCTK_VERROR("FlowTracerX ODE stage ran without CarpetX active levels");
  try {
    if (state.gf_mask)
      ensure_fields_ready(cctkGH, {state.gf_mask});
  } catch (const std::exception &error) {
    CCTK_VERROR("%s", error.what());
  }

  CarpetX::active_levels->loop_serially([&](auto &level) {
    std::array<std::array<ParticleBoundary, 3>, 2> boundaries{};
    const auto &symmetries =
        CarpetX::ghext->patchdata.at(level.patch).symmetries;
    for (int side = 0; side < 2; ++side)
      for (int d = 0; d < 3; ++d) {
        const auto symmetry = symmetries[side][d];
        boundaries[side][d] =
            symmetry == CarpetX::symmetry_t::periodic
                ? ParticleBoundary::periodic
            : symmetry == CarpetX::symmetry_t::reflection
                ? ParticleBoundary::reflect
                : ParticleBoundary::absorb;
      }
    state.containers.at(level.patch)
        ->advance_stage(level.level, transport_fields(level), boundaries);
  });
}

extern "C" void FlowTracerX_FinishStep(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!CarpetX::active_levels)
    CCTK_VERROR("FlowTracerX CCTK_POSTSTEP ran without CarpetX active levels");

  const bool levels_are_aligned = hierarchy_is_time_aligned();
  CarpetX::active_levels->loop_serially([&](auto &level) {
    auto &container = *state.containers.at(level.patch);
    // CarpetX widens active_levels at POSTSTEP to include levels that just
    // caught up. Only the batch advanced since PRESTEP has an open stage.
    if (container.stage(level.level) >= 0) {
      container.finish_step(level.level);
      if (!levels_are_aligned)
        container.redistribute_local_level(level.level,
                                           subcycling_guard_cells,
                                           max_num_cells_moved);
    }
  });

  if (levels_are_aligned) {
    for (auto &container : state.containers)
      container->redistribute_local_hierarchy(max_num_cells_moved);
    inject_due_tracer_sets(cctkGH, false);
  }
}

extern "C" void FlowTracerX_Regrid(CCTK_ARGUMENTS) {
  auto &state = runtime();
  for (auto &container : state.containers) {
    container->resize_after_regrid();
    container->redistribute_global();
  }
}

extern "C" int FlowTracerX_Shutdown() {
  runtime() = Runtime{};
  return 0;
}

} // namespace FlowTracerX

extern "C" CCTK_INT FlowTracerX_InjectParticles(
    const CCTK_INT tracer_set, const CCTK_INT count,
    const CCTK_REAL *const x, const CCTK_REAL *const y,
    const CCTK_REAL *const z, const CCTK_REAL *const weight,
    const CCTK_REAL birth_time, CCTK_INT8 *const ids) {
  using namespace FlowTracerX;
  auto &state = runtime();
  const auto communicator = amrex::ParallelDescriptor::Communicator();

  int invalid =
      !state.initialized || !hierarchy_is_time_aligned() || tracer_set < 0 ||
      tracer_set >= static_cast<CCTK_INT>(state.tracer_sets.size()) ||
      count < 0 || (count > 0 && (!x || !y || !z)) ||
      state.containers.size() != 1;
  if (!invalid && count > 0 &&
      !state.containers.front()->has_level_zero_tile())
    invalid = 1;
  int any_invalid = 0;
  MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, communicator);
  if (any_invalid)
    return -1;

  long long local_count = count;
  long long prefix = 0;
  long long total_count = 0;
  MPI_Exscan(&local_count, &prefix, 1, MPI_LONG_LONG, MPI_SUM, communicator);
  MPI_Allreduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM,
                communicator);
  if (amrex::ParallelDescriptor::MyProc() == 0)
    prefix = 0;
  if (total_count < 0 ||
      state.next_id >
          std::numeric_limits<amrex::Long>::max() - total_count)
    return -1;

  const amrex::Long first_id = state.next_id + prefix;
  state.containers.front()->inject_local(
      tracer_set, count, x, y, z, weight, ids, first_id, birth_time);
  state.next_id += total_count;
  if (total_count > 0)
    state.containers.front()->redistribute_global();
  return 0;
}

extern "C" CCTK_INT8 FlowTracerX_LocalParticleCount() {
  amrex::Long count = 0;
  for (const auto &container : FlowTracerX::runtime().containers)
    count += container->local_particle_count();
  return static_cast<CCTK_INT8>(count);
}

extern "C" CCTK_INT8 FlowTracerX_GlobalParticleCount() {
  amrex::Long count = FlowTracerX_LocalParticleCount();
  amrex::ParallelDescriptor::ReduceLongSum(count);
  return static_cast<CCTK_INT8>(count);
}

extern "C" CCTK_INT
FlowTracerX_GlobalPositionSum(CCTK_REAL position_sum[3]) {
  if (!position_sum)
    return -1;
  std::array<amrex::Real, 3> sum{{0, 0, 0}};
  for (const auto &container : FlowTracerX::runtime().containers) {
    const auto local = container->local_position_sum();
    for (int d = 0; d < 3; ++d)
      sum[d] += local[d];
  }
  amrex::ParallelDescriptor::ReduceRealSum(sum.data(), 3);
  for (int d = 0; d < 3; ++d)
    position_sum[d] = sum[d];
  return 0;
}
