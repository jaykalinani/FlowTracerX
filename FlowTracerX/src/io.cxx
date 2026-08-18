/**
 * \file io.cxx
 * \brief Parallel diagnostics and CarpetX-scheduled particle checkpointing.
 */
#include "runtime.hxx"

#include "interpolation.hxx"
#include "seeding.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleUtil.H>

#include <mpi.h>

#ifdef HAVE_CAPABILITY_openPMD_api
#include <openPMD/openPMD.hpp>
#endif

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FlowTracerX {
namespace {

constexpr int checkpoint_schema = 1;
constexpr const char *particle_dataset_name = "flowtracerx_particles";
int last_checkpoint_iteration = -1;
int last_checkpoint_runtime = -1;

using HostParticleTile =
    amrex::ParticleTile<ParticleType, RealIdx::count, IntIdx::count,
                        amrex::PolymorphicArenaAllocator>;

std::string iteration_string(const int iteration) {
  std::ostringstream stream;
  stream << std::setw(8) << std::setfill('0') << iteration;
  return stream.str();
}

std::string checkpoint_root(const std::string &directory,
                            const std::string &file, const int iteration) {
  return directory + "/" + file + ".it" + iteration_string(iteration) +
         ".flowtracerx";
}

std::string patch_name(const std::size_t patch) {
  std::ostringstream stream;
  stream << "patch" << std::setw(2) << std::setfill('0') << patch;
  return stream.str();
}

std::string record_name(std::string name) {
  for (char &c : name)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
      c = '_';
  return name;
}

void write_checkpoint(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  auto &state = runtime();
  const int iteration = cctkGH->cctk_iteration;
  if (iteration == state.recovered_iteration ||
      iteration <= last_checkpoint_iteration)
    return;
  for (const auto &container : state.containers)
    for (int level = 0; level <= container->finestLevel(); ++level)
      if (container->stage(level) >= 0)
        CCTK_VERROR("FlowTracerX cannot checkpoint an in-flight ODE stage");

  // CarpetX owns cadence and recovery policy but does not expose its internal
  // openPMD/Silo series. Use the same IO directory, file stem, iteration, and
  // schedule decision for an AMReX-native sibling checkpoint.
  const std::string root =
      checkpoint_root(checkpoint_dir, checkpoint_file, iteration);
  const int error = CCTK_CreateDirectory(0755, root.c_str());
  if (error < 0)
    CCTK_VERROR("Could not create FlowTracerX checkpoint directory '%s'",
                root.c_str());

  const amrex::Vector<std::string> real_names = {"weight", "birth_time"};
  const amrex::Vector<std::string> int_names = {"tracer_set"};
  for (std::size_t patch = 0; patch < state.containers.size(); ++patch)
    state.containers[patch]->Checkpoint(root + "/" + patch_name(patch),
                                        particle_dataset_name, real_names,
                                        int_names);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream metadata(root + "/FlowTracerX.meta");
    if (!metadata)
      CCTK_VERROR("Could not write FlowTracerX checkpoint metadata");
    metadata << "schema " << checkpoint_schema << '\n';
    metadata << "iteration " << iteration << '\n';
    metadata << std::setprecision(17) << "time " << cctkGH->cctk_time
             << '\n';
    metadata << "patches " << state.containers.size() << '\n';
    metadata << "next_id " << state.next_id << '\n';
    metadata << "tracer_config_hash " << tracer_configuration_hash() << '\n';
    metadata << "tracer_sets " << state.tracer_sets.size() << '\n';
    for (std::size_t tracer_set = 0; tracer_set < state.tracer_sets.size();
         ++tracer_set)
      metadata << "injections " << tracer_set << ' '
               << state.tracer_sets[tracer_set].injections << '\n';
  }
  amrex::ParallelDescriptor::Barrier();
  last_checkpoint_iteration = iteration;
}

void write_debug_tsv(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (debug_tsv_every <= 0 ||
      cctkGH->cctk_iteration % debug_tsv_every != 0 ||
      !hierarchy_is_time_aligned())
    return;
  auto &state = runtime();
  amrex::Long local = 0;
  for (const auto &container : state.containers)
    local += container->local_particle_count();
  amrex::Long global = local;
  amrex::ParallelDescriptor::ReduceLongSum(global);
  if (global > debug_max_particles)
    CCTK_VERROR("FlowTracerX debug TSV is limited to %d particles, but the "
                "current population is %lld",
                debug_max_particles, static_cast<long long>(global));

  if (CCTK_CreateDirectory(0755, out_dir) < 0)
    CCTK_VERROR("Could not create FlowTracerX output directory '%s'", out_dir);
  std::ostringstream filename;
  filename << out_dir << '/' << particle_dataset_name << ".it"
           << iteration_string(cctkGH->cctk_iteration) << ".rank"
           << std::setw(6) << std::setfill('0')
           << amrex::ParallelDescriptor::MyProc() << ".tsv";
  std::ofstream output(filename.str());
  if (!output)
    CCTK_VERROR("Could not create FlowTracerX debug file '%s'",
                filename.str().c_str());
  output << "# x y z id cpu tracer_set weight birth_time\n";
  output << std::setprecision(17);
  // TSV intentionally copies to the host and is therefore guarded by a hard
  // global particle limit. It is a debugging aid, never the scaling path.
  for (const auto &container : state.containers)
    for (int lev = 0; lev <= container->finestLevel(); ++lev)
      for (const auto &particle_tile : container->GetParticles(lev)) {
        HostParticleTile host_tile;
        host_tile.define(container->NumRuntimeRealComps(),
                         container->NumRuntimeIntComps(), nullptr, nullptr,
                         amrex::The_Pinned_Arena());
        host_tile.resize(particle_tile.second.numParticles());
        amrex::copyParticles(host_tile, particle_tile.second);
        const auto particles = host_tile.getConstParticleTileData();
        for (int i = 0; i < host_tile.numParticles(); ++i)
          if (particles.id(i).is_valid())
            output << particles.pos(0, i) << ' ' << particles.pos(1, i) << ' '
                   << particles.pos(2, i) << ' '
                   << static_cast<long long>(particles.id(i)) << ' '
                   << static_cast<int>(particles.cpu(i)) << ' '
                   << particles.idata(IntIdx::tracer_set)[i] << ' '
                   << particles.rdata(RealIdx::weight)[i] << ' '
                   << particles.rdata(RealIdx::birth_time)[i] << '\n';
      }
}

#ifdef HAVE_CAPABILITY_openPMD_api

template <typename T>
std::shared_ptr<T> host_array(const std::size_t size) {
  auto *const arena = amrex::The_Pinned_Arena();
  auto *const data = static_cast<T *>(arena->alloc(size * sizeof(T)));
  return std::shared_ptr<T>(data, [arena](T *const ptr) { arena->free(ptr); });
}

template <typename T>
void define_component(openPMD::RecordComponent &component,
                      const std::uint64_t count) {
  component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(),
                                          openPMD::Extent{count}));
}

void write_openpmd(const cGH *const cctkGH) {
  DECLARE_CCTK_PARAMETERS;
  if (out_every <= 0 || cctkGH->cctk_iteration % out_every != 0 ||
      !hierarchy_is_time_aligned())
    return;

  if (CCTK_CreateDirectory(0755, out_dir) < 0)
    throw std::runtime_error("Could not create the FlowTracerX output "
                             "directory");
  const std::string filename =
      std::string(out_dir) + "/" + particle_dataset_name + ".it" +
      iteration_string(cctkGH->cctk_iteration) + "." + openpmd_backend;
  openPMD::Series series(filename, openPMD::Access::CREATE,
                         amrex::ParallelDescriptor::Communicator());
  series.setSoftware("FlowTracerX", "0.1.0");
  auto &iteration = series.iterations[cctkGH->cctk_iteration];
  iteration.setAttribute("schemaVersion", checkpoint_schema);
  iteration.setAttribute("cactusTime", static_cast<double>(cctkGH->cctk_time));
  iteration.setAttribute("timeIntegrator",
                         std::string(configured_ode_method()));
  iteration.setAttribute("coordinateSystem", std::string("Cartesian"));

  auto &state = runtime();
  std::string tracer_names;
  for (int tracer_set = 0; tracer_set < num_tracer_sets; ++tracer_set) {
    if (!tracer_names.empty())
      tracer_names += ',';
    tracer_names += tracer_name[tracer_set][0] != '\0'
                        ? tracer_name[tracer_set]
                        : "tracer_" + std::to_string(tracer_set);
  }
  iteration.setAttribute("tracerNames", tracer_names);
  for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
    auto &container = *state.containers[patch];
    const long long local = container.local_particle_count();
    long long offset = 0;
    long long global = 0;
    const auto communicator = amrex::ParallelDescriptor::Communicator();
    MPI_Exscan(&local, &offset, 1, MPI_LONG_LONG, MPI_SUM, communicator);
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, communicator);
    if (amrex::ParallelDescriptor::MyProc() == 0)
      offset = 0;
    if (global == 0)
      continue;

    auto &species =
        iteration.particles[std::string(particle_dataset_name) + "_" +
                            patch_name(patch)];
    auto &position = species["position"];
    auto &position_offset = species["positionOffset"];
    for (const char *axis : {"x", "y", "z"}) {
      define_component<amrex::Real>(position[axis], global);
      define_component<amrex::Real>(position_offset[axis], global);
      position_offset[axis].makeConstant(amrex::Real(0));
    }
    auto &id = species["id"][openPMD::RecordComponent::SCALAR];
    auto &cpu = species["cpu"][openPMD::RecordComponent::SCALAR];
    auto &tracer_set =
        species["tracerSet"][openPMD::RecordComponent::SCALAR];
    auto &weight = species["weighting"][openPMD::RecordComponent::SCALAR];
    auto &birth = species["birthTime"][openPMD::RecordComponent::SCALAR];
    define_component<std::int64_t>(id, global);
    define_component<std::int32_t>(cpu, global);
    define_component<std::int32_t>(tracer_set, global);
    define_component<amrex::Real>(weight, global);
    define_component<amrex::Real>(birth, global);
    for (const auto &sample : state.sample_gfs)
      define_component<amrex::Real>(
          species[record_name(sample.output_name)]
                 [openPMD::RecordComponent::SCALAR],
          global);

    std::uint64_t current = static_cast<std::uint64_t>(offset);
    for (int level = 0; level <= container.finestLevel(); ++level) {
      const auto &level_data =
          CarpetX::ghext->patchdata.at(patch).leveldata.at(level);
      const auto prob_lo = container.Geom(level).ProbLoArray();
      const auto inv_dx = container.Geom(level).InvCellSizeArray();
      for (ParIter pti(container, level); pti.isValid(); ++pti) {
        const int count = pti.numParticles();
        if (count == 0)
          continue;
        const openPMD::Offset chunk_offset{current};
        const openPMD::Extent chunk_extent{static_cast<std::uint64_t>(count)};

        HostParticleTile host_tile;
        host_tile.define(container.NumRuntimeRealComps(),
                         container.NumRuntimeIntComps(), nullptr, nullptr,
                         amrex::The_Pinned_Arena());
        host_tile.resize(count);
        amrex::copyParticles(host_tile, pti.GetParticleTile());
        const auto particles = host_tile.getConstParticleTileData();
        auto px = host_array<amrex::Real>(count);
        auto py = host_array<amrex::Real>(count);
        auto pz = host_array<amrex::Real>(count);
        auto ids = host_array<std::int64_t>(count);
        auto cpus = host_array<std::int32_t>(count);
        auto tracer_sets = host_array<std::int32_t>(count);
        auto weights = host_array<amrex::Real>(count);
        auto births = host_array<amrex::Real>(count);
        for (int i = 0; i < count; ++i) {
          px.get()[i] = particles.pos(0, i);
          py.get()[i] = particles.pos(1, i);
          pz.get()[i] = particles.pos(2, i);
          ids.get()[i] = static_cast<std::int64_t>(particles.id(i));
          cpus.get()[i] = static_cast<std::int32_t>(particles.cpu(i));
          tracer_sets.get()[i] =
              particles.idata(IntIdx::tracer_set)[i];
          weights.get()[i] = particles.rdata(RealIdx::weight)[i];
          births.get()[i] = particles.rdata(RealIdx::birth_time)[i];
        }
        position["x"].storeChunk(px, chunk_offset, chunk_extent);
        position["y"].storeChunk(py, chunk_offset, chunk_extent);
        position["z"].storeChunk(pz, chunk_offset, chunk_extent);
        id.storeChunk(ids, chunk_offset, chunk_extent);
        cpu.storeChunk(cpus, chunk_offset, chunk_extent);
        tracer_set.storeChunk(tracer_sets, chunk_offset, chunk_extent);
        weight.storeChunk(weights, chunk_offset, chunk_extent);
        birth.storeChunk(births, chunk_offset, chunk_extent);

        // Particle fields are gathered on the GPU. Only the completed output
        // chunk is copied to pinned host memory for openPMD transport.
        const auto device_particles =
            pti.GetParticleTile().getConstParticleTileData();
        amrex::Gpu::DeviceVector<amrex::Real> device_sample(count);
        for (const auto &sample : state.sample_gfs) {
          const auto &mfab = field_mfab(level_data, sample);
          const auto values = mfab.const_array(pti);
          const auto box = mfab[pti].box();
          FieldCentering centering;
          const auto index_type = field_nodal_flags(level_data, sample);
          for (int d = 0; d < 3; ++d)
            centering.nodal[d] = index_type[d];
          amrex::Real *const destination = device_sample.dataPtr();
          const int component = sample.component_index;
          const amrex::Real nan =
              std::numeric_limits<amrex::Real>::quiet_NaN();
          amrex::ParallelFor(count,
                             [=] AMREX_GPU_DEVICE(const int i) noexcept {
            const amrex::GpuArray<amrex::Real, 3> point{
                device_particles.pos(0, i), device_particles.pos(1, i),
                device_particles.pos(2, i)};
            amrex::Real value = nan;
            trilinear_interpolate(values, component, point, prob_lo, inv_dx,
                                  centering, box, value);
            destination[i] = value;
          });
          auto host_sample = host_array<amrex::Real>(count);
          amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_sample.begin(),
                           device_sample.end(), host_sample.get());
          species[record_name(sample.output_name)]
                 [openPMD::RecordComponent::SCALAR]
              .storeChunk(host_sample, chunk_offset, chunk_extent);
        }
        current += count;
      }
      // One collective flush per AMR level bounds deferred pinned storage
      // without requiring every rank to own the same number of particle tiles.
      series.flush();
    }
  }
  iteration.close();
  series.flush();
  series.close();
}

#endif

} // namespace

extern "C" void FlowTracerX_Output(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  try {
    write_debug_tsv(cctkGH);
#ifdef HAVE_CAPABILITY_openPMD_api
    DECLARE_CCTK_PARAMETERS;
    if (out_every > 0 && cctk_iteration % out_every == 0 &&
        hierarchy_is_time_aligned())
      ensure_fields_ready(cctkGH, runtime().sample_gfs);
    write_openpmd(cctkGH);
#endif
  } catch (const std::exception &error) {
    CCTK_VERROR("FlowTracerX output failed: %s", error.what());
  }
}

extern "C" void FlowTracerX_CheckpointInitial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (checkpoint_ID) {
    write_checkpoint(cctkGH);
    last_checkpoint_runtime = CCTK_RunTime();
  }
}

extern "C" void FlowTracerX_Checkpoint(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  int run_time = CCTK_RunTime();
  MPI_Bcast(&run_time, 1, MPI_INT, 0,
            amrex::ParallelDescriptor::Communicator());
  const bool by_iteration =
      checkpoint_every > 0 && cctk_iteration % checkpoint_every == 0;
  const bool by_wall_time = checkpoint_every_walltime_hours > 0 &&
                            run_time >= last_checkpoint_runtime +
                                            std::lrint(
                                                checkpoint_every_walltime_hours *
                                                3600);
  if (by_iteration || by_wall_time) {
    write_checkpoint(cctkGH);
    last_checkpoint_runtime = run_time;
  }
}

extern "C" void FlowTracerX_CheckpointTerminate(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (checkpoint_on_terminate)
    write_checkpoint(cctkGH);
}

extern "C" void FlowTracerX_Recover(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(recover, "no"))
    return;
  auto &state = runtime();
  const std::string root =
      checkpoint_root(recover_dir, recover_file, cctk_iteration);
  std::ifstream metadata(root + "/FlowTracerX.meta");
  if (!metadata)
    CCTK_VERROR("FlowTracerX recovery sidecar '%s' is missing", root.c_str());

  int schema = -1;
  int iteration = -1;
  double time = 0;
  std::size_t patches = 0;
  std::size_t tracer_sets = 0;
  std::uint64_t config_hash = 0;
  std::string key;
  while (metadata >> key) {
    if (key == "schema")
      metadata >> schema;
    else if (key == "iteration")
      metadata >> iteration;
    else if (key == "time")
      metadata >> time;
    else if (key == "patches")
      metadata >> patches;
    else if (key == "next_id")
      metadata >> state.next_id;
    else if (key == "tracer_config_hash")
      metadata >> config_hash;
    else if (key == "tracer_sets")
      metadata >> tracer_sets;
    else if (key == "injections") {
      std::size_t tracer_set = 0;
      long count = 0;
      metadata >> tracer_set >> count;
      if (tracer_set < state.tracer_sets.size())
        state.tracer_sets[tracer_set].injections = count;
    } else {
      std::string ignored;
      std::getline(metadata, ignored);
    }
  }
  if (schema != checkpoint_schema || iteration != cctk_iteration ||
      patches != state.containers.size() ||
      tracer_sets != state.tracer_sets.size() ||
      config_hash != tracer_configuration_hash())
    CCTK_VERROR("FlowTracerX checkpoint metadata is incompatible with this "
                "run");

  for (std::size_t patch = 0; patch < state.containers.size(); ++patch) {
    state.containers[patch]->Restart(root + "/" + patch_name(patch),
                                     particle_dataset_name);
    state.containers[patch]->redistribute_global();
  }
  state.recovered = true;
  state.recovered_iteration = cctk_iteration;
  last_checkpoint_iteration = cctk_iteration;
}

} // namespace FlowTracerX
