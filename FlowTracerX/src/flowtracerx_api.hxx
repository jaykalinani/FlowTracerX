/**
 * \file flowtracerx_api.hxx
 * \brief Small C interface for optional particle-provider thorns.
 */
#ifndef FLOWTRACERX_API_HXX
#define FLOWTRACERX_API_HXX

#include <cctk.h>

extern "C" {

/**
 * Inject rank-local positions into a configured tracer set at synchronized time.
 *
 * Every rank must call this collective, including ranks with `count == 0`.
 * FlowTracerX assigns globally unique IDs and performs one AMReX hierarchy
 * redistribution. `ids` may be null. Returns zero on success and -1 for an
 * invalid call.
 */
CCTK_INT FlowTracerX_InjectParticles(CCTK_INT tracer_set, CCTK_INT count,
                                     const CCTK_REAL *x,
                                     const CCTK_REAL *y,
                                     const CCTK_REAL *z,
                                     const CCTK_REAL *weight,
                                     CCTK_REAL birth_time,
                                     CCTK_INT8 *ids);

/// Return the valid particle count owned by this MPI rank.
CCTK_INT8 FlowTracerX_LocalParticleCount();
/// Collectively return the valid particle count over all ranks.
CCTK_INT8 FlowTracerX_GlobalParticleCount();

/// Collectively sum valid particle coordinates (primarily for analytic tests).
CCTK_INT FlowTracerX_GlobalPositionSum(CCTK_REAL position_sum[3]);

}

#endif
