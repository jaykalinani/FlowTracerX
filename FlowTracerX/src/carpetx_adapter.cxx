/**
 * \file carpetx_adapter.cxx
 * \brief Checked access to CarpetX group storage and synchronization state.
 */
#include "carpetx_adapter.hxx"

#include <cctk.h>

#include <set>
#include <stdexcept>

namespace FlowTracerX {

FieldRef resolve_field(const std::string &name, const bool allow_empty) {
  if (name.empty()) {
    if (allow_empty)
      return {};
    throw std::runtime_error("FlowTracerX: an empty grid-variable name was "
                             "provided");
  }

  const int vi = CCTK_VarIndex(name.c_str());
  if (vi < 0)
    throw std::runtime_error("FlowTracerX: unknown grid variable \"" + name +
                             "\"");
  const int gi = CCTK_GroupIndexFromVarI(vi);
  if (gi < 0)
    throw std::runtime_error("FlowTracerX: cannot find the group for \"" +
                             name + "\"");

  cGroup group;
  if (CCTK_GroupData(gi, &group) != 0)
    throw std::runtime_error("FlowTracerX: cannot inspect the group for \"" +
                             name + "\"");
  if (group.grouptype != CCTK_GF || group.vartype != CCTK_VARIABLE_REAL ||
      group.dim != 3)
    throw std::runtime_error("FlowTracerX: \"" + name +
                             "\" must be a three-dimensional CCTK_REAL grid "
                             "function");

  const int first = CCTK_FirstVarIndexI(gi);
  const std::size_t separator = name.rfind("::");
  const std::string output_name =
      separator == std::string::npos ? name : name.substr(separator + 2);
  return FieldRef{vi, gi, vi - first, name, output_name};
}

const CarpetX::GHExt::PatchData::LevelData::GroupData &
group_data(const CarpetX::GHExt::PatchData::LevelData &level,
           const FieldRef &field) {
  if (!field ||
      field.group_index >= static_cast<int>(level.groupdata.size()) ||
      !level.groupdata.at(field.group_index))
    throw std::runtime_error("FlowTracerX: grid storage is unavailable for \"" +
                             field.name + "\"");
  return *level.groupdata.at(field.group_index);
}

const amrex::MultiFab &field_mfab(
    const CarpetX::GHExt::PatchData::LevelData &level, const FieldRef &field,
    const int timelevel) {
  const auto &gd = group_data(level, field);
  if (timelevel < 0 || timelevel >= static_cast<int>(gd.mfab.size()) ||
      !gd.mfab.at(timelevel))
    throw std::runtime_error("FlowTracerX: time level storage is unavailable "
                             "for \"" +
                             field.name + "\"");
  return *gd.mfab.at(timelevel);
}

std::array<int, 3>
field_nodal_flags(const CarpetX::GHExt::PatchData::LevelData &level,
                  const FieldRef &field) {
  const auto &carpetx_centering = group_data(level, field).indextype;
  std::array<int, 3> nodal;
  for (int d = 0; d < 3; ++d)
    // CarpetX: 0=vertex, 1=cell. AMReX: 1=node, 0=cell.
    nodal[d] = 1 - carpetx_centering[d];
  return nodal;
}

void ensure_fields_ready(const cGH *const cctkGH,
                         const std::vector<FieldRef> &fields) {
  if (fields.empty())
    return;
  if (!CarpetX::active_levels)
    throw std::runtime_error(
        "FlowTracerX: no active CarpetX levels are available for field sync");

  std::set<int> groups_to_sync;
  for (const auto &field : fields) {
    if (!field)
      continue;
    bool needs_sync = false;
    CarpetX::active_levels->loop_serially([&](const auto &level) {
      const auto &gd = group_data(level, field);
      const auto valid = gd.valid.at(0).at(field.component_index).get();
      if (!valid.valid_int)
        throw std::runtime_error("FlowTracerX: interior data are invalid for \"" +
                                 field.name + "\"");
      needs_sync |= !valid.valid_outer || !valid.valid_ghosts;
    });
    if (needs_sync)
      groups_to_sync.insert(field.group_index);
  }

  if (!groups_to_sync.empty()) {
    const std::vector<int> groups(groups_to_sync.begin(), groups_to_sync.end());
    const int synced = CarpetX::SyncGroupsByDirI(
        cctkGH, static_cast<int>(groups.size()), groups.data(), nullptr);
    if (synced < 0)
      throw std::runtime_error(
          "FlowTracerX: CarpetX could not synchronize a sampled field");
  }
}

bool hierarchy_is_time_aligned() { return CarpetX::all_levels_synchronized(); }

} // namespace FlowTracerX
