/**
 * \file test_flowtracerx.cxx
 * \brief Analytic GPU fields and lifecycle assertions for FlowTracerX tests.
 */
#include <loop_device.hxx>

#include <flowtracerx_api.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestFlowTracerX {
using namespace Loop;

extern "C" void TestFlowTracerX_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestFlowTracerX_Initial;
  grid.loop_all_device<1, 1, 1>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      state(p.I) = 0;
                                    });
}

extern "C" void TestFlowTracerX_SetFields(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestFlowTracerX_SetFields;
  DECLARE_CCTK_PARAMETERS;

  // Cactus parameter storage is host-only. Copy values into ordinary scalars
  // before capturing them in a performance-portable device kernel.
  const CCTK_REAL velocity_x = test_velocity[0];
  const CCTK_REAL velocity_y = test_velocity[1];
  const CCTK_REAL velocity_z = test_velocity[2];
  const CCTK_REAL lapse = test_lapse;
  const CCTK_REAL shift_x = test_shift[0];
  const CCTK_REAL shift_y = test_shift[1];
  const CCTK_REAL shift_z = test_shift[2];
  const CCTK_REAL excision_radius = mask_radius;

  grid.loop_all_device<1, 1, 1>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      velx(p.I) = velocity_x;
                                      vely(p.I) = velocity_y;
                                      velz(p.I) = velocity_z;
                                      // The spherical zero region emulates an
                                      // excision/validity mask supplied by a
                                      // production evolution thorn.
                                      const CCTK_REAL r2 =
                                          p.x * p.x + p.y * p.y + p.z * p.z;
                                      mask(p.I) =
                                          excision_radius >= 0 &&
                                                  r2 <= excision_radius *
                                                            excision_radius
                                              ? 0.0
                                              : 1.0;
                                    });
  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        alp(p.I) = lapse;
        betax(p.I) = shift_x;
        betay(p.I) = shift_y;
        betaz(p.I) = shift_z;
      });
}

extern "C" void TestFlowTracerX_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestFlowTracerX_RHS;
  grid.loop_all_device<1, 1, 1>(grid.nghostzones,
                                [=] CCTK_DEVICE(const PointDesc &p)
                                    CCTK_ATTRIBUTE_ALWAYS_INLINE {
                                      state_rhs(p.I) = 0;
                                    });
}

extern "C" void TestFlowTracerX_SyncState(CCTK_ARGUMENTS) {
  // CarpetX performs the SYNC operation declared in schedule.ccl. This empty
  // routine follows the TestODESolvers convention and provides the schedule
  // point that restores outer-boundary and ghost-zone validity after each RK
  // update.
}

extern "C" void TestFlowTracerX_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS_TestFlowTracerX_Check;
  DECLARE_CCTK_PARAMETERS;
  if (expected_particles >= 0) {
    const CCTK_INT8 count = FlowTracerX_GlobalParticleCount();
    if (count != expected_particles)
      CCTK_VERROR("Expected %d FlowTracerX particles, found %lld",
                  expected_particles, static_cast<long long>(count));
  }
  if (check_mean_position || check_linear_mean_position) {
    const CCTK_INT8 count = FlowTracerX_GlobalParticleCount();
    if (count <= 0)
      CCTK_VERROR("Cannot check the FlowTracerX mean position with no "
                  "particles");
    CCTK_REAL position_sum[3];
    if (FlowTracerX_GlobalPositionSum(position_sum) != 0)
      CCTK_VERROR("FlowTracerX position reduction failed");
    for (int d = 0; d < 3; ++d) {
      const CCTK_REAL mean = position_sum[d] / count;
      const CCTK_REAL expected =
          check_linear_mean_position
              ? initial_mean_position[d] +
                    expected_coordinate_velocity[d] * cctk_time
              : expected_mean_position[d];
      if (std::abs(mean - expected) > position_tolerance)
        CCTK_VERROR("FlowTracerX mean coordinate %d: expected %.17g, found "
                    "%.17g",
                    d, static_cast<double>(expected),
                    static_cast<double>(mean));
    }
  }
}

} // namespace TestFlowTracerX
