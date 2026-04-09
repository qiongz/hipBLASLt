#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <uopc/uopc.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <time.h>
#include <vector>

namespace py = pybind11;

namespace c10d {
namespace {

struct CollectiveScopeContext {
  bool valid = false;
  uopcParallelAxis_t parallel_axis = UOPC_AXIS_UNKNOWN;
  uopcPhase_t phase = UOPC_PHASE_UNKNOWN;
  uopcProcessGroupKind_t process_group_kind = UOPC_PG_UNKNOWN;
  uopcOverlapWindowType_t overlap_window_type = UOPC_WINDOW_UNKNOWN;
  uint64_t bucket_or_microbatch_id = 0;
  uint64_t flags = 0;
};

thread_local CollectiveScopeContext g_collective_scope{};
thread_local std::vector<CollectiveScopeContext> g_collective_scope_stack{};

bool parse_env_bool(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "off") != 0 &&
      std::strcmp(value, "no") != 0;
}

bool debug_enabled() {
  return parse_env_bool("TORCH_UOPC_PATCH_DEBUG", false);
}

const char* debug_rank() {
  const char* value = std::getenv("RANK");
  return (value != nullptr && *value != '\0') ? value : "?";
}

const char* debug_local_rank() {
  const char* value = std::getenv("LOCAL_RANK");
  return (value != nullptr && *value != '\0') ? value : "?";
}

uint64_t monotonic_timestamp_ns();

void sync_framework_scope_with_uopc() {
  (void)uopcInitialize();
  const auto pop_status = uopcPopFrameworkScope();
  if (!g_collective_scope.valid) {
    if (debug_enabled()) {
      std::fprintf(
          stderr,
          "[torch_uopc_patch] scope action=clear status=%u rank=%s local_rank=%s depth=0 ts_ns=%llu\n",
          static_cast<unsigned>(pop_status),
          debug_rank(),
          debug_local_rank(),
          static_cast<unsigned long long>(monotonic_timestamp_ns()));
    }
    return;
  }

  UopcFrameworkScopeV1 scope {};
  scope.version = UOPC_ABI_VERSION;
  scope.parallelAxis = g_collective_scope.parallel_axis;
  scope.phase = g_collective_scope.phase;
  scope.processGroupKind = g_collective_scope.process_group_kind;
  scope.overlapWindowType = g_collective_scope.overlap_window_type;
  scope.bucketOrMicrobatchId = g_collective_scope.bucket_or_microbatch_id;
  scope.flags = g_collective_scope.flags;

  const auto status = uopcPushFrameworkScope(&scope);
  if (debug_enabled()) {
    std::fprintf(
        stderr,
        "[torch_uopc_patch] scope action=sync status=%u rank=%s local_rank=%s axis=%u phase=%u pg=%u window=%u bucket=%llu flags=%llu depth=%zu ts_ns=%llu\n",
        static_cast<unsigned>(status),
        debug_rank(),
        debug_local_rank(),
        static_cast<unsigned>(scope.parallelAxis),
        static_cast<unsigned>(scope.phase),
        static_cast<unsigned>(scope.processGroupKind),
        static_cast<unsigned>(scope.overlapWindowType),
        static_cast<unsigned long long>(scope.bucketOrMicrobatchId),
        static_cast<unsigned long long>(scope.flags),
        g_collective_scope_stack.size() + 1,
        static_cast<unsigned long long>(monotonic_timestamp_ns()));
  }
  if (debug_enabled() && status != UOPC_STATUS_SUCCESS &&
      status != UOPC_STATUS_UNAVAILABLE) {
    std::fprintf(
        stderr,
        "[torch_uopc_patch] framework_scope_sync_failed status=%u axis=%u phase=%u pg=%u window=%u bucket=%llu\n",
        static_cast<unsigned>(status),
        static_cast<unsigned>(scope.parallelAxis),
        static_cast<unsigned>(scope.phase),
        static_cast<unsigned>(scope.processGroupKind),
        static_cast<unsigned>(scope.overlapWindowType),
        static_cast<unsigned long long>(scope.bucketOrMicrobatchId));
  }
}

uint64_t monotonic_timestamp_ns() {
  struct timespec ts {};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
      static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t tensor_bytes(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return static_cast<uint64_t>(tensor.numel()) *
      static_cast<uint64_t>(tensor.element_size());
}

template <typename Container>
const at::Tensor* first_defined_tensor(const Container& tensors) {
  for (const auto& tensor : tensors) {
    if (tensor.defined()) {
      return &tensor;
    }
  }
  return nullptr;
}

uint64_t total_bytes(const std::vector<at::Tensor>& tensors) {
  uint64_t bytes = 0;
  for (const auto& tensor : tensors) {
    bytes += tensor_bytes(tensor);
  }
  return bytes;
}

uint64_t total_bytes(const std::vector<std::vector<at::Tensor>>& tensors) {
  uint64_t bytes = 0;
  for (const auto& tensor_list : tensors) {
    bytes += total_bytes(tensor_list);
  }
  return bytes;
}

int tensor_device(const at::Tensor* tensor) {
  if (tensor == nullptr || !tensor->defined() || !tensor->is_cuda()) {
    return -1;
  }
  return tensor->get_device();
}

uint32_t parse_env_u32(const char* name, uint32_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return fallback;
  }
  return static_cast<uint32_t>(parsed);
}

uint32_t estimate_nchannels(uint64_t bytes, int world_size) {
  const auto env_override =
      parse_env_u32("TORCH_UOPC_ESTIMATE_NCHANNELS", 0);
  if (env_override > 0) {
    return env_override;
  }
  if (world_size <= 1) {
    return 1;
  }
  if (bytes <= (1ull << 20)) {
    return 1;
  }
  if (bytes <= (16ull << 20)) {
    return std::max<uint32_t>(1, std::min<uint32_t>(4, world_size));
  }
  if (bytes <= (64ull << 20)) {
    return std::max<uint32_t>(4, std::min<uint32_t>(32, world_size * 4));
  }
  return 56;
}

CollectiveScopeContext infer_scope(uopcCommType_t comm_type) {
  if (g_collective_scope.valid) {
    return g_collective_scope;
  }

  CollectiveScopeContext scope;
  scope.valid = true;
  switch (comm_type) {
    case UOPC_COMM_ALLREDUCE:
      scope.parallel_axis = UOPC_AXIS_DP;
      scope.phase = UOPC_PHASE_POST_BACKWARD;
      scope.process_group_kind = UOPC_PG_DP_REPLICA;
      scope.overlap_window_type = UOPC_WINDOW_BACKWARD_AR_VS_NEXT_GEMM;
      return scope;
    case UOPC_COMM_REDUCE_SCATTER:
      scope.parallel_axis = UOPC_AXIS_FSDP;
      scope.phase = UOPC_PHASE_POST_BACKWARD;
      scope.process_group_kind = UOPC_PG_DP_SHARD;
      scope.overlap_window_type = UOPC_WINDOW_BACKWARD_RS_VS_NEXT_GRAD;
      return scope;
    case UOPC_COMM_ALLGATHER:
      scope.parallel_axis = UOPC_AXIS_FSDP;
      scope.phase = UOPC_PHASE_FORWARD;
      scope.process_group_kind = UOPC_PG_DP_SHARD;
      scope.overlap_window_type = UOPC_WINDOW_FORWARD_AG_VS_CURRENT_GEMM;
      return scope;
    case UOPC_COMM_ALL_TO_ALL:
      scope.parallel_axis = UOPC_AXIS_EP;
      scope.phase = UOPC_PHASE_FORWARD;
      scope.process_group_kind = UOPC_PG_EP;
      scope.overlap_window_type = UOPC_WINDOW_A2A_VS_EXPERT_GEMM;
      return scope;
    case UOPC_COMM_BROADCAST:
    case UOPC_COMM_P2P:
    case UOPC_COMM_UNKNOWN:
      return scope;
  }
  return scope;
}

uint64_t infer_comm_hash(const c10::intrusive_ptr<Backend>& backend) {
  if (!backend) {
    return 0;
  }
  return static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(backend.get()));
}

void maybe_submit_launch(
    const c10::intrusive_ptr<Backend>& backend,
    uopcCommType_t comm_type,
    const at::Tensor* representative_tensor,
    uint64_t bytes) {
  const int device = tensor_device(representative_tensor);
  if (device < 0 || bytes == 0 || !backend) {
    return;
  }

  RcclLaunchHintV1 hint {};
  hint.version = UOPC_ABI_VERSION;
  hint.valid = 1;
  hint.device = device;
  hint.streamUid = 0;
  hint.commHash = infer_comm_hash(backend);
  hint.seqNumber = backend->getSequenceNumberForGroup();
  hint.commType = comm_type;
  hint.topologyScope = UOPC_TOPO_INTRA_NODE;
  hint.bytes = bytes;
  hint.asyncMode = 1;
  hint.nChannels = estimate_nchannels(bytes, backend->getSize());
  hint.gridX = hint.nChannels;
  hint.launchTimestampNs = monotonic_timestamp_ns();

  const auto scope = infer_scope(comm_type);
  hint.parallelAxis = scope.parallel_axis;
  hint.phase = scope.phase;
  hint.processGroupKind = scope.process_group_kind;
  hint.bucketOrMicrobatchId = scope.bucket_or_microbatch_id;

  (void)uopcInitialize();
  const auto status = uopcSubmitRcclLaunch(&hint);
  if (debug_enabled()) {
    std::fprintf(
        stderr,
        "[torch_uopc_patch] submit rank=%s local_rank=%s comm=%u seq=%llu bytes=%llu nch=%u status=%u axis=%u phase=%u pg=%u window=%u bucket=%llu stream=%llu ts_ns=%llu\n",
        debug_rank(),
        debug_local_rank(),
        static_cast<unsigned>(comm_type),
        static_cast<unsigned long long>(hint.seqNumber),
        static_cast<unsigned long long>(hint.bytes),
        hint.nChannels,
        static_cast<unsigned>(status),
        static_cast<unsigned>(hint.parallelAxis),
        static_cast<unsigned>(hint.phase),
        static_cast<unsigned>(hint.processGroupKind),
        static_cast<unsigned>(scope.overlap_window_type),
        static_cast<unsigned long long>(scope.bucket_or_microbatch_id),
        static_cast<unsigned long long>(hint.streamUid),
        static_cast<unsigned long long>(hint.launchTimestampNs));
  }
}

} // namespace

class ProcessGroupUOPCNCCL final : public Backend {
 public:
  explicit ProcessGroupUOPCNCCL(const c10::intrusive_ptr<Backend>& backend)
      : Backend(backend->getRank(), backend->getSize()), backend_(backend) {
    backend_->setSequenceNumberForGroup();
  }

  const std::string getBackendName() const override {
    return "uopc_nccl";
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return backend_->getBackendOptions();
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    backend_->setTimeout(timeout);
  }

  void setSequenceNumberForGroup() override {
    backend_->setSequenceNumberForGroup();
  }

  uint64_t getSequenceNumberForGroup() override {
    return backend_->getSequenceNumberForGroup();
  }

  void startCoalescing() override {
    backend_->startCoalescing();
  }

  c10::intrusive_ptr<Work> endCoalescing() override {
    return backend_->endCoalescing();
  }

  void abort() override {
    backend_->abort();
  }

  void shutdown() override {
    backend_->shutdown();
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    maybe_submit_launch(
        backend_, UOPC_COMM_BROADCAST, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->broadcast(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    maybe_submit_launch(
        backend_, UOPC_COMM_ALLREDUCE, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    maybe_submit_launch(
        backend_, UOPC_COMM_ALLREDUCE, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->allreduce_coalesced(tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    return backend_->reduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    const auto* tensor = first_defined_tensor(input_tensors);
    const auto bytes = std::max(total_bytes(input_tensors), total_bytes(output_tensors));
    maybe_submit_launch(backend_, UOPC_COMM_ALLGATHER, tensor, bytes);
    return backend_->allgather(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    const auto bytes = std::max(tensor_bytes(input_buffer), tensor_bytes(output_buffer));
    maybe_submit_launch(backend_, UOPC_COMM_ALLGATHER, &input_buffer, bytes);
    return backend_->_allgather_base(output_buffer, input_buffer, opts);
  }

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_tensor_lists,
      std::vector<at::Tensor>& input_tensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    const auto* tensor = first_defined_tensor(input_tensors);
    const auto bytes =
        std::max(total_bytes(input_tensors), total_bytes(output_tensor_lists));
    maybe_submit_launch(backend_, UOPC_COMM_ALLGATHER, tensor, bytes);
    return backend_->allgather_coalesced(output_tensor_lists, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    const auto* tensor = first_defined_tensor(input_tensors);
    const auto bytes = std::max(total_bytes(input_tensors), total_bytes(output_tensors));
    maybe_submit_launch(backend_, UOPC_COMM_ALLGATHER, tensor, bytes);
    return backend_->allgather_into_tensor_coalesced(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const GatherOptions& opts = GatherOptions()) override {
    return backend_->gather(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    return backend_->scatter(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    const auto* tensor = first_defined_tensor(output_tensors);
    const auto bytes =
        std::max(total_bytes(output_tensors), total_bytes(input_tensors));
    maybe_submit_launch(backend_, UOPC_COMM_REDUCE_SCATTER, tensor, bytes);
    return backend_->reduce_scatter(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    const auto bytes = std::max(tensor_bytes(input_buffer), tensor_bytes(output_buffer));
    maybe_submit_launch(backend_, UOPC_COMM_REDUCE_SCATTER, &output_buffer, bytes);
    return backend_->_reduce_scatter_base(output_buffer, input_buffer, opts);
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    const auto* tensor = first_defined_tensor(output_tensors);
    const auto bytes = std::max(total_bytes(input_tensors), total_bytes(output_tensors));
    maybe_submit_launch(backend_, UOPC_COMM_REDUCE_SCATTER, tensor, bytes);
    return backend_->reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts);
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    (void)output_split_sizes;
    (void)input_split_sizes;
    const auto bytes = std::max(tensor_bytes(input_tensor), tensor_bytes(output_tensor));
    maybe_submit_launch(backend_, UOPC_COMM_ALL_TO_ALL, &input_tensor, bytes);
    return backend_->alltoall_base(
        output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts);
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    const auto* tensor = first_defined_tensor(input_tensors);
    const auto bytes = std::max(total_bytes(input_tensors), total_bytes(output_tensors));
    maybe_submit_launch(backend_, UOPC_COMM_ALL_TO_ALL, tensor, bytes);
    return backend_->alltoall(output_tensors, input_tensors, opts);
  }

  void monitoredBarrier(
      const BarrierOptions& opts,
      bool wait_all_ranks = false) override {
    backend_->monitoredBarrier(opts, wait_all_ranks);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    return backend_->barrier(opts);
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dst_rank,
      int tag) override {
    (void)dst_rank;
    (void)tag;
    maybe_submit_launch(
        backend_, UOPC_COMM_P2P, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->send(tensors, dst_rank, tag);
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int src_rank,
      int tag) override {
    (void)src_rank;
    (void)tag;
    maybe_submit_launch(
        backend_, UOPC_COMM_P2P, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->recv(tensors, src_rank, tag);
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    (void)tag;
    maybe_submit_launch(
        backend_, UOPC_COMM_P2P, first_defined_tensor(tensors), total_bytes(tensors));
    return backend_->recvAnysource(tensors, tag);
  }

 private:
  c10::intrusive_ptr<Backend> backend_;
};

c10::intrusive_ptr<Backend> create_process_group_uopc_nccl(
    const c10::intrusive_ptr<Backend>& backend) {
  return c10::make_intrusive<ProcessGroupUOPCNCCL>(backend);
}

void push_collective_scope(
    int parallel_axis,
    int phase,
    int process_group_kind,
    int overlap_window_type,
    uint64_t bucket_or_microbatch_id,
    uint64_t flags) {
  if (g_collective_scope.valid) {
    g_collective_scope_stack.push_back(g_collective_scope);
  }
  g_collective_scope.valid = true;
  g_collective_scope.parallel_axis =
      static_cast<uopcParallelAxis_t>(parallel_axis);
  g_collective_scope.phase = static_cast<uopcPhase_t>(phase);
  g_collective_scope.process_group_kind =
      static_cast<uopcProcessGroupKind_t>(process_group_kind);
  g_collective_scope.overlap_window_type =
      static_cast<uopcOverlapWindowType_t>(overlap_window_type);
  g_collective_scope.bucket_or_microbatch_id = bucket_or_microbatch_id;
  g_collective_scope.flags = flags;
  sync_framework_scope_with_uopc();
}

void pop_collective_scope() {
  if (!g_collective_scope_stack.empty()) {
    g_collective_scope = g_collective_scope_stack.back();
    g_collective_scope_stack.pop_back();
  } else {
    g_collective_scope = {};
  }
  sync_framework_scope_with_uopc();
}

} // namespace c10d

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "create_process_group_uopc_nccl",
      &c10d::create_process_group_uopc_nccl);
  m.def("push_collective_scope", &c10d::push_collective_scope);
  m.def("pop_collective_scope", &c10d::pop_collective_scope);
}
