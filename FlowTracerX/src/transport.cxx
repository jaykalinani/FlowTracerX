/**
 * \file transport.cxx
 * \brief AsterX-compatible coordinate velocity on the CarpetX mesh.
 */
#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>

#include "aster_interp.hxx"

namespace FlowTracerX {
using namespace Arith;
using namespace AsterUtils;
using namespace Loop;

extern "C" void FlowTracerX_PrepareVtilde(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_FlowTracerX_PrepareVtilde;

  const vec<GF3D2<const CCTK_REAL>, 3> gf_vel{velx, vely, velz};
  const vec<GF3D2<const CCTK_REAL>, 3> gf_beta{betax, betay, betaz};
  const vec<GF3D2<CCTK_REAL>, 3> gf_vtilde{vtilde_x, vtilde_y, vtilde_z};

  // HydroBaseX velocity is cell centered, whereas ADMBaseX lapse and shift are
  // vertex centered. Use AsterUtils::calc_avg_v2c exactly as AsterX does before
  // particles interpolate the resulting cell-centered coordinate velocity.
  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        const CCTK_REAL alp_avg = calc_avg_v2c(alp, p);
        for (int d = 0; d < 3; ++d)
          gf_vtilde(d)(p.I) =
              alp_avg * gf_vel(d)(p.I) - calc_avg_v2c(gf_beta(d), p);
      });
}

} // namespace FlowTracerX
