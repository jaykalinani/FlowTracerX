/**
 * \file carpetx_adapter.hxx
 * \brief Narrow compatibility layer between FlowTracerX and CarpetX internals.
 */
#ifndef FLOWTRACERX_CARPETX_ADAPTER_HXX
#define FLOWTRACERX_CARPETX_ADAPTER_HXX

#include <driver.hxx>
#include <schedule.hxx>

#include <AMReX_MultiFab.H>

#include <array>
#include <string>
#include <vector>

namespace FlowTracerX {

struct FieldRef {
  int var_index = -1;
  int group_index = -1;
  int component_index = -1;
  std::string name;
  std::string output_name;

  explicit operator bool() const noexcept { return var_index >= 0; }
};

/// Resolve and validate a qualified three-dimensional CCTK_REAL grid function.
FieldRef resolve_field(const std::string &name, bool allow_empty = false);

const CarpetX::GHExt::PatchData::LevelData::GroupData &
group_data(const CarpetX::GHExt::PatchData::LevelData &level,
           const FieldRef &field);

const amrex::MultiFab &field_mfab(
    const CarpetX::GHExt::PatchData::LevelData &level, const FieldRef &field,
    int timelevel = 0);

std::array<int, 3>
field_nodal_flags(const CarpetX::GHExt::PatchData::LevelData &level,
                  const FieldRef &field);

/// Synchronize dynamic fields only when their CarpetX validity requires it.
void ensure_fields_ready(const cGH *cctkGH,
                         const std::vector<FieldRef> &fields);

bool hierarchy_is_time_aligned();

} // namespace FlowTracerX

#endif
