/**
 * \file particle_container.cxx
 * \brief GPU particle push, lifecycle handling, seeding, and redistribution.
 */
#include "particle_container.hxx"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace FlowTracerX {
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE std::uint64_t
mix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
uniform01(const std::uint64_t seed, const std::uint64_t counter) noexcept {
  const std::uint64_t bits = mix64(seed ^ mix64(counter));
  return static_cast<amrex::Real>((bits >> 11U) * 0x1.0p-53);
}

/** Apply one patch's face policy before the next stage interpolation. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void apply_boundaries(
    amrex::GpuArray<amrex::Real, 3> &pos,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &prob_hi,
    const amrex::GpuArray<int, 6> &boundary,
    const bool canonicalize_periodic, bool &escaped) noexcept {
  for (int d = 0; d < 3 && !escaped; ++d) {
    const amrex::Real length = prob_hi[d] - prob_lo[d];
    if (!(length > 0)) {
      escaped = true;
      continue;
    }

    // More than a few crossings in one fluid step indicates a badly violated
    // particle CFL. The loop nevertheless handles ordinary repeated periodic
    // or reflection crossings without a branch-heavy general mapping object.
    for (int crossing = 0; crossing < 8; ++crossing) {
      if (pos[d] < prob_lo[d]) {
        const auto face = static_cast<ParticleBoundary>(boundary[2 * d]);
        if (face == ParticleBoundary::periodic) {
          // Keep intermediate RK positions in the periodic image adjacent to
          // the owning tile. Its synchronized ghost cells then provide the
          // next on-device gather without a stage-local particle exchange.
          if (!canonicalize_periodic)
            break;
          pos[d] += length *
                    (amrex::Real(1) +
                     amrex::Math::floor((prob_lo[d] - pos[d]) / length));
        } else if (face == ParticleBoundary::reflect) {
          pos[d] = amrex::Real(2) * prob_lo[d] - pos[d];
        } else {
          escaped = true;
          break;
        }
      } else if (pos[d] >= prob_hi[d]) {
        const auto face = static_cast<ParticleBoundary>(boundary[2 * d + 1]);
        if (face == ParticleBoundary::periodic) {
          if (!canonicalize_periodic)
            break;
          pos[d] -= length *
                    (amrex::Real(1) +
                     amrex::Math::floor((pos[d] - prob_hi[d]) / length));
        } else if (face == ParticleBoundary::reflect) {
          pos[d] = amrex::Real(2) * prob_hi[d] - pos[d];
          // AMReX domains are half open. Keep an exactly reflected upper-face
          // point on the representable interior side of that face.
          if (pos[d] >= prob_hi[d])
            pos[d] = prob_hi[d] -
                     amrex::Real(16) *
                         std::numeric_limits<amrex::Real>::epsilon() * length;
        } else {
          escaped = true;
          break;
        }
      } else {
        break;
      }
    }
    const bool allowed_periodic_image =
        !canonicalize_periodic &&
        ((pos[d] < prob_lo[d] &&
          static_cast<ParticleBoundary>(boundary[2 * d]) ==
              ParticleBoundary::periodic) ||
         (pos[d] >= prob_hi[d] &&
          static_cast<ParticleBoundary>(boundary[2 * d + 1]) ==
              ParticleBoundary::periodic));
    if ((pos[d] < prob_lo[d] || pos[d] >= prob_hi[d]) &&
        !allowed_periodic_image)
      escaped = true;
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool passes_lifecycle_fields(
    const amrex::GpuArray<amrex::Real, 3> &pos,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx, const bool use_alp,
    const amrex::Array4<const amrex::Real> &gf_alp, const int comp_alp,
    const FieldCentering centering_alp, const amrex::Box &box_alp,
    const amrex::Real alp_floor, const bool use_mask,
    const amrex::Array4<const amrex::Real> &gf_mask, const int comp_mask,
    const FieldCentering centering_mask, const amrex::Box &box_mask,
    const bool mask_below, const amrex::Real mask_value) noexcept {
  if (use_alp) {
    amrex::Real alp = 0;
    if (!trilinear_interpolate(gf_alp, comp_alp, pos, prob_lo, inv_dx,
                               centering_alp, box_alp, alp) ||
        alp <= alp_floor)
      return false;
  }
  if (use_mask) {
    amrex::Real mask = 0;
    if (!trilinear_interpolate(gf_mask, comp_mask, pos, prob_lo, inv_dx,
                               centering_mask, box_mask, mask))
      return false;
    if (mask_below ? mask <= mask_value : mask >= mask_value)
      return false;
  }
  return true;
}

} // namespace

ParticleContainer::ParticleContainer(amrex::AmrCore *const core)
    : BaseContainer(core) {
  SetSoACompileTimeNames({"x", "y", "z", "weight", "birth_time"},
                         {"tracer_set"});
}

ParticleContainer::LevelState &ParticleContainer::level_state(const int lev) {
  if (lev >= static_cast<int>(level_states_.size()))
    level_states_.resize(lev + 1);
  return level_states_.at(lev);
}

void ParticleContainer::begin_step(const int lev,
                                   const TimeIntegrator integrator,
                                   const amrex::Real dt) {
  auto &state = level_state(lev);
  if (state.stage >= 0)
    amrex::Abort("FlowTracerX began a particle step before the previous "
                 "Runge--Kutta sequence completed");
  state.integrator = integrator;
  state.stage = 0;
  state.dt = dt;
}

void ParticleContainer::advance_stage(
    const int lev, const TransportFields &fields,
    const std::array<std::array<ParticleBoundary, 3>, 2> &boundaries) {
  auto &state = level_state(lev);
  if (state.stage < 0)
    amrex::Abort("FlowTracerX particle stage executed without CCTK_PRESTEP");
  const int expected_stages =
      state.integrator == TimeIntegrator::rk4 ? 4 : 3;
  if (state.stage >= expected_stages)
    amrex::Abort("FlowTracerX received too many ODESolvers_RHS calls");

  const int stage = state.stage;
  const amrex::Real dt = state.dt;
  const bool use_alp = fields.alp && fields.alp_floor >= 0;
  const bool use_mask = fields.mask != nullptr;
  const auto prob_lo = Geom(lev).ProbLoArray();
  const auto prob_hi = Geom(lev).ProbHiArray();
  const auto inv_dx = Geom(lev).InvCellSizeArray();

  amrex::GpuArray<int, 6> boundary_codes;
  for (int d = 0; d < 3; ++d) {
    boundary_codes[2 * d] = static_cast<int>(boundaries[0][d]);
    boundary_codes[2 * d + 1] = static_cast<int>(boundaries[1][d]);
  }

  for (ParIter pti(*this, lev); pti.isValid(); ++pti) {
    const int np = pti.numParticles();
    if (np == 0)
      continue;

    const TileKey key{pti.index(), pti.LocalTileIndex()};
    auto &scratch = state.tile_scratch[key].rk_data;
    if (stage == 0)
      scratch.resize(6 * static_cast<std::size_t>(np));
    if (scratch.size() != 6 * static_cast<std::size_t>(np))
      amrex::Abort("FlowTracerX particle tile changed between ODE stages");

    auto particles = pti.GetParticleTile().getParticleTileData();
    amrex::Real *const rk_data = scratch.dataPtr();
    amrex::Real *const x_n = rk_data;
    amrex::Real *const y_n = rk_data + np;
    amrex::Real *const z_n = rk_data + 2 * np;
    amrex::Real *const rhs_x = rk_data + 3 * np;
    amrex::Real *const rhs_y = rk_data + 4 * np;
    amrex::Real *const rhs_z = rk_data + 5 * np;

    amrex::Array4<const amrex::Real> gf_vtilde[3];
    amrex::Box box_vtilde[3];
    for (int d = 0; d < 3; ++d) {
      gf_vtilde[d] = fields.vtilde[d]->const_array(pti);
      box_vtilde[d] = (*fields.vtilde[d])[pti].box();
    }

    amrex::Array4<const amrex::Real> gf_alp;
    amrex::Box box_alp;
    if (use_alp) {
      gf_alp = fields.alp->const_array(pti);
      box_alp = (*fields.alp)[pti].box();
    }
    amrex::Array4<const amrex::Real> gf_mask;
    amrex::Box box_mask;
    if (use_mask) {
      gf_mask = fields.mask->const_array(pti);
      box_mask = (*fields.mask)[pti].box();
    }

    const auto centering_vtilde_x = fields.vtilde_centering[0];
    const auto centering_vtilde_y = fields.vtilde_centering[1];
    const auto centering_vtilde_z = fields.vtilde_centering[2];
    const auto centering_alp = fields.alp_centering;
    const auto centering_mask = fields.mask_centering;
    const int comp_vtilde_x = fields.vtilde_component[0];
    const int comp_vtilde_y = fields.vtilde_component[1];
    const int comp_vtilde_z = fields.vtilde_component[2];
    const int comp_alp = fields.alp_component;
    const int comp_mask = fields.mask_component;
    const bool mask_below = fields.mask_below;
    const amrex::Real mask_value = fields.mask_value;
    const amrex::Real alp_floor = fields.alp_floor;
    const bool use_rk4 = state.integrator == TimeIntegrator::rk4;

    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(const int i) noexcept {
      if (!particles.id(i).is_valid())
        return;

      amrex::GpuArray<amrex::Real, 3> pos{{particles.pos(0, i),
                                           particles.pos(1, i),
                                           particles.pos(2, i)}};
      if (stage == 0) {
        x_n[i] = pos[0];
        y_n[i] = pos[1];
        z_n[i] = pos[2];
        rhs_x[i] = rhs_y[i] = rhs_z[i] = 0;
      }

      if (!passes_lifecycle_fields(
              pos, prob_lo, inv_dx, use_alp, gf_alp, comp_alp,
              centering_alp, box_alp, alp_floor, use_mask, gf_mask, comp_mask,
              centering_mask, box_mask, mask_below, mask_value)) {
        particles.id(i) = -1;
        return;
      }

      amrex::GpuArray<amrex::Real, 3> vtilde;
      if (!trilinear_interpolate(gf_vtilde[0], comp_vtilde_x, pos, prob_lo,
                                 inv_dx, centering_vtilde_x, box_vtilde[0],
                                 vtilde[0]) ||
          !trilinear_interpolate(gf_vtilde[1], comp_vtilde_y, pos, prob_lo,
                                 inv_dx, centering_vtilde_y, box_vtilde[1],
                                 vtilde[1]) ||
          !trilinear_interpolate(gf_vtilde[2], comp_vtilde_z, pos, prob_lo,
                                 inv_dx, centering_vtilde_z, box_vtilde[2],
                                 vtilde[2])) {
        particles.id(i) = -1;
        return;
      }

      // ODESolvers invokes this kernel once per explicit stage. IMEX42L uses
      // the RK4 explicit tableau; IMEX32L uses the SSPRK3 explicit tableau.
      if (use_rk4) {
        if (stage == 0) {
          rhs_x[i] = vtilde[0];
          rhs_y[i] = vtilde[1];
          rhs_z[i] = vtilde[2];
          pos = {{x_n[i] + dt * vtilde[0] / 2,
                  y_n[i] + dt * vtilde[1] / 2,
                  z_n[i] + dt * vtilde[2] / 2}};
        } else if (stage == 1) {
          rhs_x[i] += 2 * vtilde[0];
          rhs_y[i] += 2 * vtilde[1];
          rhs_z[i] += 2 * vtilde[2];
          pos = {{x_n[i] + dt * vtilde[0] / 2,
                  y_n[i] + dt * vtilde[1] / 2,
                  z_n[i] + dt * vtilde[2] / 2}};
        } else if (stage == 2) {
          rhs_x[i] += 2 * vtilde[0];
          rhs_y[i] += 2 * vtilde[1];
          rhs_z[i] += 2 * vtilde[2];
          pos = {{x_n[i] + dt * vtilde[0], y_n[i] + dt * vtilde[1],
                  z_n[i] + dt * vtilde[2]}};
        } else {
          pos = {{x_n[i] + dt * (rhs_x[i] + vtilde[0]) / 6,
                  y_n[i] + dt * (rhs_y[i] + vtilde[1]) / 6,
                  z_n[i] + dt * (rhs_z[i] + vtilde[2]) / 6}};
        }
      } else {
        if (stage == 0) {
          rhs_x[i] = vtilde[0];
          rhs_y[i] = vtilde[1];
          rhs_z[i] = vtilde[2];
          pos = {{x_n[i] + dt * vtilde[0], y_n[i] + dt * vtilde[1],
                  z_n[i] + dt * vtilde[2]}};
        } else if (stage == 1) {
          rhs_x[i] += vtilde[0];
          rhs_y[i] += vtilde[1];
          rhs_z[i] += vtilde[2];
          pos = {{x_n[i] + dt * rhs_x[i] / 4,
                  y_n[i] + dt * rhs_y[i] / 4,
                  z_n[i] + dt * rhs_z[i] / 4}};
        } else {
          pos = {{x_n[i] + dt * (rhs_x[i] / 6 + 2 * vtilde[0] / 3),
                  y_n[i] + dt * (rhs_y[i] / 6 + 2 * vtilde[1] / 3),
                  z_n[i] + dt * (rhs_z[i] / 6 + 2 * vtilde[2] / 3)}};
        }
      }

      bool escaped = false;
      apply_boundaries(pos, prob_lo, prob_hi, boundary_codes, false, escaped);
      if (escaped || !amrex::Math::isfinite(pos[0]) ||
          !amrex::Math::isfinite(pos[1]) ||
          !amrex::Math::isfinite(pos[2]) ||
          !passes_lifecycle_fields(
              pos, prob_lo, inv_dx, use_alp, gf_alp, comp_alp,
              centering_alp, box_alp, alp_floor, use_mask, gf_mask, comp_mask,
              centering_mask, box_mask, mask_below, mask_value)) {
        particles.id(i) = -1;
        return;
      }
      // Keep a periodic crossing in the image adjacent to its current tile.
      // AMReX canonicalizes that coordinate during Redistribute(). Wrapping it
      // here would look like a domain-sized jump and violate the bounded-local
      // redistribution contract used between CarpetX subcycling sync points.
      particles.pos(0, i) = pos[0];
      particles.pos(1, i) = pos[1];
      particles.pos(2, i) = pos[2];
    });
  }

  ++state.stage;
}

void ParticleContainer::finish_step(const int lev) {
  auto &state = level_state(lev);
  const int expected_stages =
      state.integrator == TimeIntegrator::rk4 ? 4 : 3;
  if (state.stage != expected_stages)
    amrex::Abort("FlowTracerX completed a particle step with the wrong number "
                 "of ODESolvers_RHS calls");
  state.stage = -1;
}

int ParticleContainer::stage(const int lev) const {
  return lev < static_cast<int>(level_states_.size())
             ? level_states_[lev].stage
             : -1;
}

bool ParticleContainer::has_level_zero_tile() const {
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
    return true;
  return false;
}

void ParticleContainer::seed_global(const SeedConfig &seed) {
  if (seed.count <= 0)
    return;

  const int rank = amrex::ParallelDescriptor::MyProc();
  const int has_tile = has_level_zero_tile() ? 1 : 0;
  int active_ranks = 0;
  int active_rank = 0;
  MPI_Allreduce(&has_tile, &active_ranks, 1, MPI_INT, MPI_SUM,
                amrex::ParallelDescriptor::Communicator());
  MPI_Exscan(&has_tile, &active_rank, 1, MPI_INT, MPI_SUM,
             amrex::ParallelDescriptor::Communicator());
  if (rank == 0)
    active_rank = 0;
  if (!has_tile)
    return;

  const amrex::Long rank_first = seed.count * active_rank / active_ranks;
  const amrex::Long rank_last =
      seed.count * (active_rank + 1) / active_ranks;
  const amrex::Long rank_count = rank_last - rank_first;
  if (rank_count == 0)
    return;

  int num_tiles = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
    ++num_tiles;

  const auto domain_lo = Geom(0).ProbLoArray();
  const auto domain_hi = Geom(0).ProbHiArray();
  amrex::GpuArray<amrex::Real, 3> center;
  amrex::GpuArray<amrex::Real, 3> box_center;
  amrex::GpuArray<amrex::Real, 3> box_size;
  amrex::GpuArray<int, 3> lattice_shape;
  for (int d = 0; d < 3; ++d) {
    center[d] = seed.center[d];
    box_size[d] = seed.box_size[d] > 0
                      ? seed.box_size[d]
                      : domain_hi[d] - domain_lo[d];
    box_center[d] = seed.box_size[d] > 0
                        ? seed.center[d]
                        : (domain_lo[d] + domain_hi[d]) / 2;
    lattice_shape[d] = seed.lattice_shape[d];
  }

  int tile_ordinal = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid();
       ++mfi, ++tile_ordinal) {
    const amrex::Long tile_begin =
        rank_count * tile_ordinal / num_tiles;
    const amrex::Long tile_end =
        rank_count * (tile_ordinal + 1) / num_tiles;
    const amrex::Long tile_count = tile_end - tile_begin;
    if (tile_count == 0)
      continue;

    auto &tile = DefineAndReturnParticleTile(0, mfi);
    const amrex::Long old_size = tile.numParticles();
    tile.resize(old_size + tile_count);
    auto particles = tile.getParticleTileData();
    const auto shape = seed.shape;
    const auto random_seed = seed.random_seed;
    const auto inner_radius = seed.inner_radius;
    const auto radius = seed.radius;
    const auto weight = seed.weight;
    const auto birth_time = seed.birth_time;
    const auto first_id = seed.first_id;
    const auto tracer_set = seed.tracer_set;

    amrex::ParallelFor(
        tile_count,
        [=] AMREX_GPU_DEVICE(const amrex::Long tile_index) noexcept {
          const amrex::Long rank_index = tile_begin + tile_index;
          const amrex::Long global_index = rank_first + rank_index;
          amrex::GpuArray<amrex::Real, 3> pos;
          if (shape == SeedConfig::Shape::lattice) {
            const amrex::Long ix = global_index % lattice_shape[0];
            const amrex::Long iy =
                (global_index / lattice_shape[0]) % lattice_shape[1];
            const amrex::Long iz =
                global_index / (lattice_shape[0] * lattice_shape[1]);
            const amrex::Long index[3] = {ix, iy, iz};
            for (int d = 0; d < 3; ++d)
              pos[d] = box_center[d] - box_size[d] / 2 +
                       (index[d] + amrex::Real(0.5)) * box_size[d] /
                           lattice_shape[d];
          } else if (shape == SeedConfig::Shape::random_box) {
            for (int d = 0; d < 3; ++d)
              pos[d] = box_center[d] - box_size[d] / 2 +
                       uniform01(random_seed + d, global_index) * box_size[d];
          } else {
            const amrex::Real u0 =
                uniform01(random_seed, 3 * global_index);
            const amrex::Real u1 =
                uniform01(random_seed, 3 * global_index + 1);
            const amrex::Real u2 =
                uniform01(random_seed, 3 * global_index + 2);
            const amrex::Real r0 =
                shape == SeedConfig::Shape::random_shell ? inner_radius : 0;
            const amrex::Real particle_radius =
                std::cbrt(r0 * r0 * r0 +
                          u0 * (radius * radius * radius - r0 * r0 * r0));
            const amrex::Real cos_theta = 2 * u1 - 1;
            const amrex::Real sin_theta = std::sqrt(
                amrex::max(amrex::Real(0), 1 - cos_theta * cos_theta));
            const amrex::Real phi =
                2 * amrex::Math::pi<amrex::Real>() * u2;
            const auto sincos_phi = amrex::Math::sincos(phi);
            pos[0] = center[0] +
                     particle_radius * sin_theta * sincos_phi.second;
            pos[1] = center[1] +
                     particle_radius * sin_theta * sincos_phi.first;
            pos[2] = center[2] + particle_radius * cos_theta;
          }

          const amrex::Long i = old_size + tile_index;
          particles.id(i) = first_id + global_index;
          particles.cpu(i) = rank;
          for (int d = 0; d < 3; ++d)
            particles.pos(d, i) = pos[d];
          particles.rdata(RealIdx::weight)[i] = weight;
          particles.rdata(RealIdx::birth_time)[i] = birth_time;
          particles.idata(IntIdx::tracer_set)[i] = tracer_set;
        });
  }
}

void ParticleContainer::inject_local(
    const int tracer_set, const int count, const CCTK_REAL *const x,
    const CCTK_REAL *const y, const CCTK_REAL *const z,
    const CCTK_REAL *const weight, CCTK_INT8 *const ids,
    const amrex::Long first_id, const amrex::Real birth_time) {
  if (count <= 0)
    return;

  amrex::Gpu::DeviceVector<amrex::Real> coordinates(3 * count);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, x, x + count,
                   coordinates.begin());
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, y, y + count,
                   coordinates.begin() + count);
  amrex::Gpu::copy(amrex::Gpu::hostToDevice, z, z + count,
                   coordinates.begin() + 2 * count);
  amrex::Gpu::DeviceVector<amrex::Real> weights;
  if (weight) {
    weights.resize(count);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, weight, weight + count,
                     weights.begin());
  }

  int num_tiles = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
    ++num_tiles;
  const int rank = amrex::ParallelDescriptor::MyProc();
  int tile_ordinal = 0;
  for (amrex::MFIter mfi = MakeMFIter(0); mfi.isValid();
       ++mfi, ++tile_ordinal) {
    const int tile_begin = count * tile_ordinal / num_tiles;
    const int tile_end = count * (tile_ordinal + 1) / num_tiles;
    const int tile_count = tile_end - tile_begin;
    if (tile_count == 0)
      continue;

    auto &tile = DefineAndReturnParticleTile(0, mfi);
    const amrex::Long old_size = tile.numParticles();
    tile.resize(old_size + tile_count);
    auto particles = tile.getParticleTileData();
    const amrex::Real *const xyz = coordinates.dataPtr();
    const amrex::Real *const particle_weight =
        weight ? weights.dataPtr() : nullptr;
    amrex::ParallelFor(tile_count,
                       [=] AMREX_GPU_DEVICE(const int tile_index) noexcept {
      const int input_index = tile_begin + tile_index;
      const amrex::Long i = old_size + tile_index;
      particles.id(i) = first_id + input_index;
      particles.cpu(i) = rank;
      particles.pos(0, i) = xyz[input_index];
      particles.pos(1, i) = xyz[count + input_index];
      particles.pos(2, i) = xyz[2 * count + input_index];
      particles.rdata(RealIdx::weight)[i] =
          particle_weight ? particle_weight[input_index] : amrex::Real(1);
      particles.rdata(RealIdx::birth_time)[i] = birth_time;
      particles.idata(IntIdx::tracer_set)[i] = tracer_set;
    });
  }

  amrex::Gpu::streamSynchronize();
  if (ids)
    for (int i = 0; i < count; ++i)
      ids[i] = static_cast<CCTK_INT8>(first_id + i);
}

void ParticleContainer::redistribute_local_level(
    const int lev, const int guard_cells, const int max_num_cells_moved) {
  // nGrow retains particles near their current fine box while the coarser
  // level is at a different time. `local` independently bounds neighbor
  // communication. This is ownership handling, not AMReX time subcycling.
  Redistribute(lev, lev, guard_cells, max_num_cells_moved, true);
}

void ParticleContainer::redistribute_local_hierarchy(
    const int max_num_cells_moved) {
  Redistribute(0, -1, 0, max_num_cells_moved, true);
}

void ParticleContainer::redistribute_global() {
  Redistribute(0, -1, 0, 0, true);
}

void ParticleContainer::resize_after_regrid() {
  resizeData();
  level_states_.clear();
}

amrex::Long ParticleContainer::local_particle_count() const {
  amrex::Long count = 0;
  for (int lev = 0; lev <= finestLevel(); ++lev)
    for (const auto &entry : GetParticles(lev))
      count += entry.second.numParticles();
  return count;
}

std::array<amrex::Real, 3> ParticleContainer::local_position_sum() const {
  // Device reduction avoids particle-sized host transfers and single-address
  // atomics under large populations.
  amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                   amrex::ReduceOpSum>
      reduce_ops;
  using ReduceData =
      amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real>;
  using ConstPTD = BaseContainer::ConstPTDType;
  const auto reduced = amrex::ParticleReduce<ReduceData>(
      *this,
      [=] AMREX_GPU_DEVICE(const ConstPTD &particles,
                           const int i) noexcept
          -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real> {
        if (!particles.id(i).is_valid())
          return {0, 0, 0};
        return {particles.pos(0, i), particles.pos(1, i),
                particles.pos(2, i)};
      },
      reduce_ops);
  return {amrex::get<0>(reduced), amrex::get<1>(reduced),
          amrex::get<2>(reduced)};
}

} // namespace FlowTracerX
