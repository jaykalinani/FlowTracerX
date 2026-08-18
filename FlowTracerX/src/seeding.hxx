#ifndef FLOWTRACERX_SEEDING_HXX
#define FLOWTRACERX_SEEDING_HXX

#include <cctk.h>

#include <cstdint>

namespace FlowTracerX {

void inject_due_tracer_sets(const cGH *cctkGH, bool initial_call);

/// Stable hash used to reject recovery with a different tracer configuration.
std::uint64_t tracer_configuration_hash();

} // namespace FlowTracerX

#endif
