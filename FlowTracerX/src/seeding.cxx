/**
 * \file seeding.cxx
 * \brief Deterministic tracer-set seeding and synchronized injection cadence.
 */
#include "seeding.hxx"

#include "runtime.hxx"

#include <cctk_Parameters.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace FlowTracerX {
namespace {

bool tracer_set_is_due(const int tracer_set, const cGH *const cctkGH,
                       const long ordinal) {
  DECLARE_CCTK_PARAMETERS;
  if (!tracer_enabled[tracer_set])
    return false;
  if (start_iteration[tracer_set] >= 0 &&
      cctkGH->cctk_iteration < start_iteration[tracer_set])
    return false;
  if (start_time[tracer_set] >= 0 &&
      cctkGH->cctk_time < start_time[tracer_set])
    return false;
  if (ordinal == 0)
    return true;
  if (injection_every_iterations[tracer_set] > 0) {
    const long start = std::max(0, start_iteration[tracer_set]);
    return cctkGH->cctk_iteration >=
           start + ordinal * injection_every_iterations[tracer_set];
  }
  if (injection_every_time[tracer_set] > 0) {
    const CCTK_REAL start =
        std::max(CCTK_REAL(0), start_time[tracer_set]);
    const CCTK_REAL target =
        start + ordinal * injection_every_time[tracer_set];
    const CCTK_REAL tolerance =
        16 * std::numeric_limits<CCTK_REAL>::epsilon() *
        std::max(CCTK_REAL(1), std::abs(target));
    return cctkGH->cctk_time + tolerance >= target;
  }
  return false;
}

SeedConfig make_seed_config(const int tracer_set, const cGH *const cctkGH,
                            const amrex::Long first_id) {
  DECLARE_CCTK_PARAMETERS;
  SeedConfig seed;
  seed.tracer_set = tracer_set;
  seed.first_id = first_id;
  seed.birth_time = cctkGH->cctk_time;
  seed.weight = particle_weight[tracer_set];
  seed.random_seed = static_cast<std::uint64_t>(random_seed[tracer_set]);
  seed.lattice_shape = {lattice_nx[tracer_set], lattice_ny[tracer_set],
                        lattice_nz[tracer_set]};
  seed.center = {seed_center_x[tracer_set], seed_center_y[tracer_set],
                 seed_center_z[tracer_set]};
  seed.box_size = {seed_box_size_x[tracer_set],
                   seed_box_size_y[tracer_set],
                   seed_box_size_z[tracer_set]};
  seed.inner_radius = seed_inner_radius[tracer_set];
  seed.radius = seed_radius[tracer_set];

  if (CCTK_EQUALS(seed_type[tracer_set], "lattice")) {
    seed.shape = SeedConfig::Shape::lattice;
    seed.count = static_cast<amrex::Long>(seed.lattice_shape[0]) *
                 seed.lattice_shape[1] * seed.lattice_shape[2];
  } else if (CCTK_EQUALS(seed_type[tracer_set], "random_box")) {
    seed.shape = SeedConfig::Shape::random_box;
    seed.count = num_particles[tracer_set];
  } else if (CCTK_EQUALS(seed_type[tracer_set], "random_sphere")) {
    seed.shape = SeedConfig::Shape::random_sphere;
    seed.count = num_particles[tracer_set];
  } else if (CCTK_EQUALS(seed_type[tracer_set], "random_shell")) {
    seed.shape = SeedConfig::Shape::random_shell;
    seed.count = num_particles[tracer_set];
  } else {
    CCTK_VERROR("FlowTracerX encountered unsupported seed_type '%s'",
                seed_type[tracer_set]);
  }
  return seed;
}

template <typename T>
void hash_bytes(std::uint64_t &hash, const T &value) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
}

void hash_text(std::uint64_t &hash, const char *const text) {
  for (const unsigned char *p =
           reinterpret_cast<const unsigned char *>(text);
       *p; ++p) {
    hash ^= *p;
    hash *= 1099511628211ULL;
  }
  hash ^= 0xffU;
  hash *= 1099511628211ULL;
}

} // namespace

std::uint64_t tracer_configuration_hash() {
  DECLARE_CCTK_PARAMETERS;
  std::uint64_t hash = 1469598103934665603ULL;
  hash_bytes(hash, num_tracer_sets);
  hash_text(hash, destroy_mask_var);
  hash_text(hash, destroy_mask_comparison);
  hash_bytes(hash, destroy_mask_value);
  hash_bytes(hash, destroy_if_lapse_below);
  for (int tracer_set = 0; tracer_set < num_tracer_sets; ++tracer_set) {
    hash_text(hash, tracer_name[tracer_set]);
    hash_bytes(hash, tracer_enabled[tracer_set]);
    hash_text(hash, seed_type[tracer_set]);
    hash_bytes(hash, num_particles[tracer_set]);
    hash_bytes(hash, lattice_nx[tracer_set]);
    hash_bytes(hash, lattice_ny[tracer_set]);
    hash_bytes(hash, lattice_nz[tracer_set]);
    hash_bytes(hash, seed_center_x[tracer_set]);
    hash_bytes(hash, seed_center_y[tracer_set]);
    hash_bytes(hash, seed_center_z[tracer_set]);
    hash_bytes(hash, seed_box_size_x[tracer_set]);
    hash_bytes(hash, seed_box_size_y[tracer_set]);
    hash_bytes(hash, seed_box_size_z[tracer_set]);
    hash_bytes(hash, seed_radius[tracer_set]);
    hash_bytes(hash, seed_inner_radius[tracer_set]);
    hash_bytes(hash, random_seed[tracer_set]);
    hash_bytes(hash, particle_weight[tracer_set]);
    hash_bytes(hash, start_iteration[tracer_set]);
    hash_bytes(hash, start_time[tracer_set]);
    hash_bytes(hash, injection_every_iterations[tracer_set]);
    hash_bytes(hash, injection_every_time[tracer_set]);
  }
  return hash;
}

void inject_due_tracer_sets(const cGH *const cctkGH,
                            const bool initial_call) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  if (!hierarchy_is_time_aligned())
    return;
  if (state.containers.size() != 1)
    CCTK_VERROR("Built-in FlowTracerX seeding requires the supported "
                "single-patch Cartesian path");

  // Global seed ordinals determine IDs and random positions. Checkpoints
  // restore both counters, preventing scheduled reinjection after recovery.
  bool inserted = false;
  for (int tracer_set = 0; tracer_set < num_tracer_sets; ++tracer_set) {
    auto &tracer_state = state.tracer_sets.at(tracer_set);
    if (!tracer_set_is_due(tracer_set, cctkGH, tracer_state.injections))
      continue;
    if (!initial_call && tracer_state.injections == 0 &&
        cctkGH->cctk_iteration == 0)
      continue;

    const SeedConfig seed =
        make_seed_config(tracer_set, cctkGH, state.next_id);
    if (seed.count > 0) {
      if (seed.first_id >
          std::numeric_limits<amrex::Long>::max() - seed.count)
        CCTK_VERROR("FlowTracerX particle ID space is exhausted");
      state.containers.front()->seed_global(seed);
      state.next_id += seed.count;
      inserted = true;
      if (verbose) {
        const char *const name = tracer_name[tracer_set][0] != '\0'
                                     ? tracer_name[tracer_set]
                                     : "unnamed";
        CCTK_VINFO("Injected %lld particles into tracer set %d (%s)",
                   static_cast<long long>(seed.count), tracer_set, name);
      }
    }
    ++tracer_state.injections;
  }

  if (inserted)
    state.containers.front()->redistribute_global();
}

} // namespace FlowTracerX
