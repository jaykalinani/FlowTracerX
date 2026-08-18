/**
 * \file particle_container.hxx
 * \brief Pure-SoA AMReX storage and GPU transport for passive tracers.
 */
#ifndef FLOWTRACERX_PARTICLE_CONTAINER_HXX
#define FLOWTRACERX_PARTICLE_CONTAINER_HXX

#include "interpolation.hxx"

#include <AMReX_AmrParticles.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

#include <cctk.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace FlowTracerX {

enum class TimeIntegrator { rk4, ssprk3 };
enum class ParticleBoundary : int { absorb = 0, periodic = 1, reflect = 2 };

struct RealIdx {
  enum { weight = AMREX_SPACEDIM, birth_time, count };
};

struct IntIdx {
  enum { tracer_set = 0, count };
};

using ParticleType = amrex::SoAParticle<RealIdx::count, IntIdx::count>;
using BaseContainer =
    amrex::AmrParticleContainer_impl<ParticleType, RealIdx::count,
                                     IntIdx::count>;
using ParIter = amrex::ParIterSoA<RealIdx::count, IntIdx::count>;

/** Grid data sampled by a particle stage; all interpolation occurs on-device. */
struct TransportFields {
  const amrex::MultiFab *vtilde[3]{nullptr, nullptr, nullptr};
  int vtilde_component[3]{0, 1, 2};
  FieldCentering vtilde_centering[3];

  const amrex::MultiFab *alp = nullptr;
  int alp_component = 0;
  FieldCentering alp_centering;

  const amrex::MultiFab *mask = nullptr;
  int mask_component = 0;
  FieldCentering mask_centering;
  bool mask_below = true;
  amrex::Real mask_value = 0;
  amrex::Real alp_floor = -1;
};

struct SeedConfig {
  enum class Shape { lattice, random_box, random_sphere, random_shell };
  Shape shape = Shape::lattice;
  amrex::Long count = 0;
  std::array<int, 3> lattice_shape{{1, 1, 1}};
  std::array<amrex::Real, 3> center{{0, 0, 0}};
  std::array<amrex::Real, 3> box_size{{0, 0, 0}};
  amrex::Real inner_radius = 0;
  amrex::Real radius = 0;
  std::uint64_t random_seed = 0;
  amrex::Real weight = 1;
  int tracer_set = 0;
  amrex::Real birth_time = 0;
  amrex::Long first_id = 1;
};

class ParticleContainer final : public BaseContainer {
public:
  /// Attach pure-SoA particle storage to one CarpetX patch hierarchy.
  explicit ParticleContainer(amrex::AmrCore *core);

  /// Begin one level-local fluid step; particles remain fixed to this level.
  void begin_step(int lev, TimeIntegrator integrator, amrex::Real dt);
  /// Interpolate stage fields and update particles in tile-local GPU kernels.
  void advance_stage(
      int lev, const TransportFields &fields,
      const std::array<std::array<ParticleBoundary, 3>, 2> &boundaries);
  /// Validate and close the current explicit Runge--Kutta stage sequence.
  void finish_step(int lev);
  int stage(int lev) const;

  /// Generate one decomposition-independent tracer set directly on the GPU.
  void seed_global(const SeedConfig &seed);
  /// Add rank-local provider particles before a collective redistribution.
  void inject_local(int tracer_set, int count, const CCTK_REAL *x,
                    const CCTK_REAL *y, const CCTK_REAL *z,
                    const CCTK_REAL *weight, CCTK_INT8 *ids,
                    amrex::Long first_id, amrex::Real birth_time);
  bool has_level_zero_tile() const;

  /// Exchange nearby particles on one subcycling level.
  void redistribute_local_level(int lev, int guard_cells,
                                int max_num_cells_moved);
  /// Re-level nearby particles when the AMR hierarchy is time aligned.
  void redistribute_local_hierarchy(int max_num_cells_moved);
  /// Perform rank-global ownership discovery after seed/regrid/recovery.
  void redistribute_global();
  /// Resize hierarchy metadata and invalidate tile-stage scratch after regrid.
  void resize_after_regrid();

  amrex::Long local_particle_count() const;
  std::array<amrex::Real, 3> local_position_sum() const;

private:
  using TileKey = std::pair<int, int>;
  struct TileScratch {
    amrex::Gpu::DeviceVector<amrex::Real> rk_data;
  };
  struct LevelState {
    TimeIntegrator integrator = TimeIntegrator::rk4;
    int stage = -1;
    amrex::Real dt = 0;
    std::map<TileKey, TileScratch> tile_scratch;
  };

  std::vector<LevelState> level_states_;
  LevelState &level_state(int lev);
};

} // namespace FlowTracerX

#endif
