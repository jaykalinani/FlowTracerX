/**
 * \file interpolation.hxx
 * \brief Device-callable interpolation of CarpetX grid functions.
 */
#ifndef FLOWTRACERX_INTERPOLATION_HXX
#define FLOWTRACERX_INTERPOLATION_HXX

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <array>

namespace FlowTracerX {

struct FieldCentering {
  amrex::GpuArray<int, 3> nodal{{0, 0, 0}};
};

/**
 * Trilinearly gather one scalar component at a particle position.
 *
 * `available` includes the local ghost region. Returning `false` instead of
 * clamping an index prevents a particle from silently sampling another AMR
 * box or an unavailable guard cell. This routine is called inside AMReX GPU
 * kernels for both transport fields and diagnostics.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
trilinear_interpolate(
    const amrex::Array4<const amrex::Real> &gf, const int component,
    const amrex::GpuArray<amrex::Real, 3> &pos,
    const amrex::GpuArray<amrex::Real, 3> &prob_lo,
    const amrex::GpuArray<amrex::Real, 3> &inv_dx,
    const FieldCentering centering, const amrex::Box &available,
    amrex::Real &value) noexcept {
  amrex::GpuArray<int, 3> cell;
  amrex::GpuArray<amrex::Real, 3> frac;
  for (int d = 0; d < 3; ++d) {
    const amrex::Real offset = centering.nodal[d] ? amrex::Real(0)
                                                  : amrex::Real(0.5);
    const amrex::Real q = (pos[d] - prob_lo[d]) * inv_dx[d] - offset;
    cell[d] = static_cast<int>(amrex::Math::floor(q));
    frac[d] = q - cell[d];
  }

  const auto lo = amrex::lbound(available);
  const auto hi = amrex::ubound(available);
  if (cell[0] < lo.x || cell[1] < lo.y || cell[2] < lo.z ||
      cell[0] + 1 > hi.x || cell[1] + 1 > hi.y || cell[2] + 1 > hi.z)
    return false;

  value = 0;
  for (int dk = 0; dk <= 1; ++dk) {
    const amrex::Real wk = dk ? frac[2] : amrex::Real(1) - frac[2];
    for (int dj = 0; dj <= 1; ++dj) {
      const amrex::Real wj = dj ? frac[1] : amrex::Real(1) - frac[1];
      for (int di = 0; di <= 1; ++di) {
        const amrex::Real wi = di ? frac[0] : amrex::Real(1) - frac[0];
        value += wi * wj * wk *
                 gf(cell[0] + di, cell[1] + dj, cell[2] + dk, component);
      }
    }
  }
  return amrex::Math::isfinite(value);
}

} // namespace FlowTracerX

#endif
