/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#include <libloaderapi.h>

// Remove defines that conflict locally.
#undef CONST
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#endif

#include "UserDrivenTuningParser.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocblaslt.h"
#include "rocblaslt_mat_utils.hpp"
#include "rocroller_host.hpp"
#include "tensile_host.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#ifdef HIPBLASLT_ENABLE_UOPC
#include <uopc/uopc.h>
#endif

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

namespace
{
    constexpr uint32_t kUopcHintSourceController = 1;
    constexpr uint32_t kUopcHintSourcePreference = 2;

    inline void recordRuntimeHintTelemetry(bool success, bool controllerSource);

    inline bool overrideTelemetryEnabled()
    {
        static const bool enabled = [] {
            const auto* rawValue = std::getenv("HIPBLASLT_OVERRIDE_TELEMETRY");
            return rawValue != nullptr && rawValue[0] != '\0' && std::strcmp(rawValue, "0") != 0
                   && std::strcmp(rawValue, "false") != 0 && std::strcmp(rawValue, "off") != 0
                   && std::strcmp(rawValue, "no") != 0;
        }();
        return enabled;
    }

    inline uint64_t overrideTelemetryFlushCalls()
    {
        static const uint64_t flushCalls = [] {
            const auto* rawValue = std::getenv("HIPBLASLT_OVERRIDE_TELEMETRY_FLUSH_CALLS");
            if(rawValue == nullptr || rawValue[0] == '\0')
                return uint64_t(0);

            char*      endPtr = nullptr;
            const auto value  = std::strtoull(rawValue, &endPtr, 10);
            if(endPtr == rawValue || (endPtr != nullptr && *endPtr != '\0'))
                return uint64_t(0);
            return static_cast<uint64_t>(value);
        }();
        return flushCalls;
    }

    inline const char* overrideTelemetryRank()
    {
        const auto* rank = std::getenv("RANK");
        return rank != nullptr && rank[0] != '\0' ? rank : "?";
    }

    inline const char* overrideTelemetryLocalRank()
    {
        const auto* localRank = std::getenv("LOCAL_RANK");
        return localRank != nullptr && localRank[0] != '\0' ? localRank : "?";
    }

#ifdef HIPBLASLT_ENABLE_UOPC
    inline bool uopcDebugEnabled()
    {
        const auto* rawValue = std::getenv("UOPC_DUMP_HINTS");
        return rawValue != nullptr && rawValue[0] != '\0' && std::strcmp(rawValue, "0") != 0
               && std::strcmp(rawValue, "false") != 0 && std::strcmp(rawValue, "off") != 0;
    }

    inline bool uopcEnabled()
    {
        static const bool enabled = [] {
            const auto* rawValue = std::getenv("UOPC_ENABLE");
            return rawValue == nullptr || rawValue[0] == '\0'
                   || (std::strcmp(rawValue, "0") != 0 && std::strcmp(rawValue, "false") != 0
                       && std::strcmp(rawValue, "off") != 0 && std::strcmp(rawValue, "no") != 0);
        }();
        return enabled;
    }

    inline bool isBackwardOverlapWindow(uopcOverlapWindowType_t window)
    {
        switch(window)
        {
        case UOPC_WINDOW_BACKWARD_AR_VS_NEXT_GEMM:
        case UOPC_WINDOW_BACKWARD_RS_VS_NEXT_GRAD:
        case UOPC_WINDOW_BACKWARD_PREFETCH_AG_VS_CURRENT_GRAD:
            return true;
        default:
            return false;
        }
    }

    inline const char* runtimeHintRejectReason(const GemmOverlapHintV1& hint, uint32_t visibleCuCount)
    {
        if(hint.source != UOPC_HINT_SOURCE_CONTROLLER)
            return "non_controller_source";
        if(hint.relatedSeqNumber == 0)
            return "missing_related_seq";
        if(!isBackwardOverlapWindow(hint.overlapWindowType))
            return "non_backward_window";
        if(hint.mode == UOPC_HINT_MODE_NONE)
            return "mode_none";
        if(hint.effectiveCuUpperBound == 0)
            return "missing_effective_cu";
        if(visibleCuCount > 0 && hint.effectiveCuUpperBound >= visibleCuCount)
            return "not_reduced_cu";
        return nullptr;
    }

    inline void dumpRuntimeHintTrace(const char*                        path,
                                     const RocblasltContractionProblem& prob,
                                     const GemmHintQueryV1&            query,
                                     const GemmOverlapHintV1*          hint,
                                     uopcStatus_t                      status,
                                     bool                              accepted,
                                     const char*                       reason,
                                     uint32_t                          visibleCuCount)
    {
        if(!uopcDebugEnabled())
            return;

        std::fprintf(stderr,
                     "[hipblaslt][uopc-trace] rank=%s local_rank=%s event=hint path=%s "
                     "accepted=%u reason=%s status=%u source=%u mode=%u bucket=%u upper=%u "
                     "pressure=%u window=%u topo=%u related_seq=%llu stream=%llu grouped=%u "
                     "m=%llu n=%llu k=%llu visible_cu=%u\n",
                     overrideTelemetryRank(),
                     overrideTelemetryLocalRank(),
                     path,
                     accepted ? 1U : 0U,
                     reason != nullptr ? reason : "none",
                     static_cast<unsigned>(status),
                     hint != nullptr ? static_cast<unsigned>(hint->source) : 0U,
                     hint != nullptr ? static_cast<unsigned>(hint->mode) : 0U,
                     hint != nullptr ? hint->effectiveCuBucket : 0U,
                     hint != nullptr ? hint->effectiveCuUpperBound : 0U,
                     hint != nullptr ? static_cast<unsigned>(hint->commPressureLevel) : 0U,
                     hint != nullptr ? static_cast<unsigned>(hint->overlapWindowType) : 0U,
                     hint != nullptr ? static_cast<unsigned>(hint->topologyScope) : 0U,
                     hint != nullptr ? static_cast<unsigned long long>(hint->relatedSeqNumber) : 0ULL,
                     static_cast<unsigned long long>(query.streamUid),
                     prob.grouped_gemm ? 1U : 0U,
                     static_cast<unsigned long long>(prob.m),
                     static_cast<unsigned long long>(prob.n),
                     static_cast<unsigned long long>(prob.k),
                     visibleCuCount);
    }
#endif

    inline bool hasUopcPreference(const rocblaslt_matmul_preference pref)
    {
        return pref != nullptr
               && (pref->overlap_mode || pref->effective_cu_count || pref->effective_cu_bucket
                   || pref->comm_pressure_level || pref->topology_scope
                   || pref->policy_hint_version);
    }

    inline void applyUopcPreferenceToProblem(const rocblaslt_matmul_preference pref,
                                             RocblasltContractionProblem&      prob)
    {
        if(!hasUopcPreference(pref))
            return;

        prob.uopc.valid             = 1;
        prob.uopc.mode              = pref->overlap_mode;
        prob.uopc.effectiveCuCount  = pref->effective_cu_count;
        prob.uopc.effectiveCuBucket = pref->effective_cu_bucket;
        prob.uopc.commPressureLevel = pref->comm_pressure_level;
        prob.uopc.topologyScope     = pref->topology_scope;
        prob.uopc.source            = kUopcHintSourcePreference;
    }

#ifdef HIPBLASLT_ENABLE_UOPC
    inline void applyRuntimeUopcHint(rocblaslt_handle handle, RocblasltContractionProblem& prob)
    {
        if(prob.uopc.valid)
            return;
        if(!uopcEnabled())
            return;

        GemmHintQueryV1   query{};
        GemmOverlapHintV1 hint{};

        query.version        = UOPC_ABI_VERSION;
        query.device         = handle->device;
        query.streamUid      = 0;
        query.computeBackend = UOPC_BACKEND_HIPBLASLT;
        query.problemClass
            = prob.grouped_gemm ? UOPC_PROBLEM_GROUPED_GEMM : UOPC_PROBLEM_GEMM;
        query.m = prob.m;
        query.n = prob.n;
        query.k = prob.k;
        query.semanticFlags = UOPC_GEMM_SEMANTIC_NONE;
        if(prob.gradient)
            query.semanticFlags |= UOPC_GEMM_SEMANTIC_GRADIENT_EPILOGUE;
        if(prob.grouped_gemm)
            query.semanticFlags |= UOPC_GEMM_SEMANTIC_GROUPED;
        if(prob.trans_a != HIPBLAS_OP_N)
            query.semanticFlags |= UOPC_GEMM_SEMANTIC_TRANS_A;
        if(prob.trans_b != HIPBLAS_OP_N)
            query.semanticFlags |= UOPC_GEMM_SEMANTIC_TRANS_B;

        const auto status = uopcQueryGemmHint(&query, &hint);
        const auto visibleCuCount
            = handle != nullptr && handle->properties.multiProcessorCount > 0
                  ? static_cast<uint32_t>(handle->properties.multiProcessorCount)
                  : 0U;
        if(status == UOPC_STATUS_SUCCESS && hint.valid)
        {
            recordRuntimeHintTelemetry(true, hint.source == UOPC_HINT_SOURCE_CONTROLLER);
            const auto rejectReason = runtimeHintRejectReason(hint, visibleCuCount);
            const bool acceptHint   = rejectReason == nullptr;
            dumpRuntimeHintTrace("heuristic",
                                 prob,
                                 query,
                                 &hint,
                                 status,
                                 acceptHint,
                                 acceptHint ? "accepted" : rejectReason,
                                 visibleCuCount);
            if(acceptHint)
            {
                prob.uopc.valid               = 1;
                prob.uopc.mode                = hint.mode;
                prob.uopc.effectiveCuCount    = hint.effectiveCuUpperBound;
                prob.uopc.effectiveCuBucket   = hint.effectiveCuBucket;
                prob.uopc.commPressureLevel   = hint.commPressureLevel;
                prob.uopc.overlapWindowType   = hint.overlapWindowType;
                prob.uopc.topologyScope       = hint.topologyScope;
                prob.uopc.source              = hint.source;
                prob.uopc.relatedSeqNumber    = hint.relatedSeqNumber;

                if(uopcDebugEnabled())
                {
                    std::fprintf(stderr,
                                 "[hipblaslt][uopc] heuristic runtime hint source=%u bucket=%u upper=%u "
                                 "pressure=%u seq=%llu stream=%llu\n",
                                 static_cast<unsigned>(hint.source),
                                 hint.effectiveCuBucket,
                                 hint.effectiveCuUpperBound,
                                 static_cast<unsigned>(hint.commPressureLevel),
                                 static_cast<unsigned long long>(hint.relatedSeqNumber),
                                 static_cast<unsigned long long>(query.streamUid));
                }
            }
        }
        else
        {
            recordRuntimeHintTelemetry(false, false);
            dumpRuntimeHintTrace("heuristic",
                                 prob,
                                 query,
                                 nullptr,
                                 status,
                                 false,
                                 status == UOPC_STATUS_UNAVAILABLE ? "no_snapshot"
                                                                   : "query_failed",
                                 visibleCuCount);
        }
    }
#else
    inline void applyRuntimeUopcHint(rocblaslt_handle, RocblasltContractionProblem&)
    {
    }
#endif

    inline std::string normalizeOverrideArchName(const char* gcnArchName)
    {
        if(gcnArchName == nullptr)
            return "";

        std::string arch(gcnArchName);
        const auto  pos = arch.find(':');
        if(pos != std::string::npos)
            arch.erase(pos);

        return arch;
    }

    inline std::string getOverrideArchName(rocblaslt_handle handle)
    {
        return handle != nullptr ? normalizeOverrideArchName(handle->properties.gcnArchName) : "";
    }

    inline uint32_t getVisibleCuCount(rocblaslt_handle handle)
    {
        return handle != nullptr && handle->properties.multiProcessorCount > 0
                   ? static_cast<uint32_t>(handle->properties.multiProcessorCount)
                   : 0;
    }

    inline uint32_t getOverrideCuCount(rocblaslt_handle              handle,
                                       const RocblasltContractionProblem& problem)
    {
        return (problem.uopc.valid && problem.uopc.effectiveCuCount > 0)
                   ? problem.uopc.effectiveCuCount
                   : getVisibleCuCount(handle);
    }

    inline void clearProblemUopcForRuntimeFallback(RocblasltContractionProblem& problem)
    {
        problem.uopc.valid               = 0;
        problem.uopc.effectiveCuCount    = 0;
        problem.uopc.effectiveCuBucket   = 0;
        problem.uopc.commPressureLevel   = 0;
        problem.uopc.overlapWindowType   = 0;
        problem.uopc.topologyScope       = 0;
        problem.uopc.relatedSeqNumber    = 0;
    }

    inline void setOverrideAlgoStoredEffectiveCuCount(rocblaslt_matmul_algo& algo,
                                                      uint32_t               effectiveCuCount)
    {
        std::memcpy(algo.data + sizeof(uint32_t), &effectiveCuCount, sizeof(uint32_t));
    }

    inline TensileLite::ProblemOverride
        makeOverrideLookupKey(const TensileLite::ProblemOverride& baseKey,
                              const std::string&                 arch    = "",
                              uint32_t                           cuCount = 0)
    {
        return TensileLite::ProblemOverride(baseKey.transA(),
                                            baseKey.transB(),
                                            baseKey.inputTypeA(),
                                            baseKey.inputTypeB(),
                                            baseKey.computeType(),
                                            baseKey.outputType(),
                                            baseKey.m(),
                                            baseKey.n(),
                                            baseKey.k(),
                                            baseKey.batchSize(),
                                            arch,
                                            cuCount,
                                            baseKey.biasVector(),
                                            baseKey.biasType(),
                                            baseKey.auxType(),
                                            baseKey.activationType());
    }

    enum class OverrideLookupTier : uint8_t
    {
        none = 0,
        exact,
        arch_default,
        legacy,
    };

    inline const char* toString(OverrideLookupTier tier)
    {
        switch(tier)
        {
        case OverrideLookupTier::none:
            return "none";
        case OverrideLookupTier::exact:
            return "exact";
        case OverrideLookupTier::arch_default:
            return "arch_default";
        case OverrideLookupTier::legacy:
            return "legacy";
        }

        return "unknown";
    }

    struct CachedOverrideResult
    {
        bool                              initialized   = false;
        bool                              success       = false;
        int                               solutionIndex = -1;
        uint32_t                          resolvedCuCount = 0;
        OverrideLookupTier                tier          = OverrideLookupTier::none;
        rocblaslt_matmul_heuristic_result result{};
    };

    class OverrideHeuristicCache
    {
    public:
        static OverrideHeuristicCache& instance()
        {
            static OverrideHeuristicCache gInstance;
            return gInstance;
        }

        bool lookup(const std::string& cacheKey, CachedOverrideResult& cached)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto                  it = m_cache.find(cacheKey);
            if(it == m_cache.end())
                return false;
            cached = it->second;
            return true;
        }

        void store(const std::string& cacheKey, const CachedOverrideResult& cached)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_cache[cacheKey] = cached;
        }

    private:
        std::mutex                             m_mutex;
        std::map<std::string, CachedOverrideResult> m_cache;
    };

    inline std::string makeOverrideCacheKey(const TensileLite::ProblemOverride& baseKey,
                                            size_t                              maxWorkspaceBytes)
    {
        std::ostringstream os;
        os << baseKey.arch() << '|'
           << baseKey.cuCount() << '|'
           << baseKey.transA() << '|'
           << baseKey.transB() << '|'
           << static_cast<int>(baseKey.inputTypeA()) << '|'
           << static_cast<int>(baseKey.inputTypeB()) << '|'
           << static_cast<int>(baseKey.computeType()) << '|'
           << static_cast<int>(baseKey.outputType()) << '|'
           << baseKey.m() << '|'
           << baseKey.n() << '|'
           << baseKey.k() << '|'
           << baseKey.batchSize() << '|'
           << baseKey.activationType() << '|'
           << baseKey.biasVector() << '|'
           << static_cast<int>(baseKey.biasType()) << '|'
           << static_cast<int>(baseKey.auxType()) << '|'
           << maxWorkspaceBytes;
        return os.str();
    }

    inline std::string makeScopedOverrideCacheKey(const TensileLite::ProblemOverride& baseKey,
                                                  size_t                              maxWorkspaceBytes)
    {
        return std::string("scoped|") + makeOverrideCacheKey(baseKey, maxWorkspaceBytes);
    }

    inline bool shouldUseScopedUopcExact(const RocblasltContractionProblem& problem)
    {
        return problem.uopc.valid && problem.uopc.source == kUopcHintSourceController
               && problem.uopc.effectiveCuCount > 0;
    }

    enum class OverrideApiPath : uint8_t
    {
        c_api = 0,
        cpp_api,
    };

    inline const char* toString(OverrideApiPath path)
    {
        switch(path)
        {
        case OverrideApiPath::c_api:
            return "c";
        case OverrideApiPath::cpp_api:
            return "cpp";
        }

        return "unknown";
    }

#ifdef HIPBLASLT_ENABLE_UOPC
    inline void dumpOverrideTrace(OverrideApiPath                     path,
                                  const char*                         reason,
                                  const TensileLite::ProblemOverride& baseKey,
                                  OverrideLookupTier                  tier,
                                  uint32_t                            requestedCuCount,
                                  uint32_t                            resolvedCuCount,
                                  uint32_t                            visibleCuCount,
                                  bool                                cacheHit,
                                  bool                                success,
                                  int                                 solutionIndex,
                                  uint32_t                            hintSource,
                                  uint32_t                            overlapWindowType,
                                  uint64_t                            relatedSeqNumber)
    {
        if(!uopcDebugEnabled())
            return;

        std::fprintf(stderr,
                     "[hipblaslt][uopc-trace] rank=%s local_rank=%s event=override path=%s "
                     "success=%u cache_hit=%u reason=%s tier=%s requested_cu=%u resolved_cu=%u "
                     "visible_cu=%u "
                     "source=%u window=%u related_seq=%llu solution_index=%d m=%zu n=%zu k=%zu "
                     "b=%zu act=%s bias=%d\n",
                     overrideTelemetryRank(),
                     overrideTelemetryLocalRank(),
                     toString(path),
                     success ? 1U : 0U,
                     cacheHit ? 1U : 0U,
                     reason != nullptr ? reason : "none",
                     toString(tier),
                     requestedCuCount,
                     resolvedCuCount,
                     visibleCuCount,
                     hintSource,
                     overlapWindowType,
                     static_cast<unsigned long long>(relatedSeqNumber),
                     solutionIndex,
                     baseKey.m(),
                     baseKey.n(),
                     baseKey.k(),
                     baseKey.batchSize(),
                     baseKey.activationType().c_str(),
                     baseKey.biasVector());
    }
#else
    inline void dumpOverrideTrace(OverrideApiPath,
                                  const char*,
                                  const TensileLite::ProblemOverride&,
                                  OverrideLookupTier,
                                  uint32_t,
                                  uint32_t,
                                  uint32_t,
                                  bool,
                                  bool,
                                  int,
                                  uint32_t,
                                  uint32_t,
                                  uint64_t)
    {
    }
#endif

    struct OverrideLookupOutcome
    {
        OverrideLookupTier tier = OverrideLookupTier::none;
        std::vector<int>   candidates{};
        uint32_t           resolvedCuCount = 0;
        size_t             exactCount       = 0;
        size_t             archDefaultCount = 0;
        size_t             legacyCount      = 0;
    };

    class OverrideTelemetry
    {
    public:
        static OverrideTelemetry& instance()
        {
            static OverrideTelemetry gInstance;
            return gInstance;
        }

        ~OverrideTelemetry()
        {
            dumpSummary("final");
        }

        void recordCacheLookup(OverrideApiPath                     path,
                               const TensileLite::ProblemOverride& baseKey,
                               const std::string&                 cacheKey,
                               bool                               hit)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            auto&                       stats = touchStatsLocked(baseKey);
            ++m_totalCalls;
            ++stats.calls;
            if(path == OverrideApiPath::c_api)
                ++m_cApiCalls;
            else
                ++m_cppApiCalls;
            m_uniqueCacheKeys.insert(cacheKey);
            if(hit)
            {
                ++m_cacheHits;
                ++stats.cacheHits;
            }
            else
            {
                ++m_cacheMisses;
                ++stats.cacheMisses;
            }
            maybeDumpPeriodicLocked();
        }

        void recordLookupTier(const TensileLite::ProblemOverride& baseKey, OverrideLookupTier tier)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            auto&                       stats = touchStatsLocked(baseKey);
            switch(tier)
            {
            case OverrideLookupTier::none:
                ++m_noMatchLookups;
                ++stats.noMatchLookups;
                break;
            case OverrideLookupTier::exact:
                ++m_exactLookups;
                ++stats.exactLookups;
                break;
            case OverrideLookupTier::arch_default:
                ++m_archDefaultLookups;
                ++stats.archDefaultLookups;
                break;
            case OverrideLookupTier::legacy:
                ++m_legacyLookups;
                ++stats.legacyLookups;
                break;
            }
        }

        void recordRuntimeHintOutcome(bool success, bool controllerSource)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            if(success)
            {
                if(controllerSource)
                    ++m_runtimeControllerHits;
                else
                    ++m_runtimeNonControllerHits;
            }
            else
            {
                ++m_runtimeNoSnapshot;
            }
        }

        void recordRuntime304Fallback()
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_runtime304Fallbacks;
        }

        void recordCandidateAttempt(const TensileLite::ProblemOverride& baseKey,
                                    int                                solutionIndex,
                                    bool                               accepted)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            auto&                       stats = touchStatsLocked(baseKey);
            ++m_candidateAttempts;
            ++stats.candidateAttempts;
            if(!accepted)
            {
                ++m_candidateRejects;
                ++stats.candidateRejects;
                if(solutionIndex >= 0)
                    ++m_rejectedSolutionCounts[solutionIndex];
            }
        }

        void recordFinalResult(const TensileLite::ProblemOverride& baseKey,
                               OverrideLookupTier                 tier,
                               bool                               success,
                               int                                solutionIndex)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            auto&                       stats = touchStatsLocked(baseKey);
            if(success)
            {
                ++m_successes;
                ++stats.successes;
                if(baseKey.cuCount() == 272 && tier == OverrideLookupTier::exact)
                    ++m_exact272Selected;
                if(baseKey.cuCount() == 304)
                    ++m_fallback304Selected;
                if(solutionIndex >= 0)
                    ++m_selectedSolutionCounts[solutionIndex];
            }
            else
            {
                ++m_failures;
                ++stats.failures;
            }
        }

        void recordDurations(uint64_t totalNs, uint64_t preloadNs, uint64_t cacheProbeNs, uint64_t lookupNs)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            m_totalNs += totalNs;
            m_preloadNs += preloadNs;
            m_cacheProbeNs += cacheProbeNs;
            m_lookupNs += lookupNs;
        }

    private:
        struct KeyStats
        {
            uint64_t calls              = 0;
            uint64_t cacheHits          = 0;
            uint64_t cacheMisses        = 0;
            uint64_t successes          = 0;
            uint64_t failures           = 0;
            uint64_t exactLookups       = 0;
            uint64_t archDefaultLookups = 0;
            uint64_t legacyLookups      = 0;
            uint64_t noMatchLookups     = 0;
            uint64_t candidateAttempts  = 0;
            uint64_t candidateRejects   = 0;
        };

        std::string makeProblemKey(const TensileLite::ProblemOverride& baseKey) const
        {
            std::ostringstream os;
            os << "arch=" << baseKey.arch() << "|cu=" << baseKey.cuCount() << "|ta="
               << (baseKey.transA() ? 'T' : 'N') << "|tb=" << (baseKey.transB() ? 'T' : 'N')
               << "|m=" << baseKey.m() << "|n=" << baseKey.n() << "|k=" << baseKey.k()
               << "|b=" << baseKey.batchSize() << "|act=" << baseKey.activationType()
               << "|bias=" << baseKey.biasVector();
            return os.str();
        }

        KeyStats& touchStatsLocked(const TensileLite::ProblemOverride& baseKey)
        {
            const auto problemKey = makeProblemKey(baseKey);
            m_uniqueProblemKeys.insert(problemKey);
            return m_keyStats[problemKey];
        }

        template <typename Key, typename Value>
        static std::vector<std::pair<Key, Value>>
            sortCountsDescending(const std::map<Key, Value>& values, size_t limit)
        {
            std::vector<std::pair<Key, Value>> entries(values.begin(), values.end());
            std::sort(entries.begin(),
                      entries.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
            if(entries.size() > limit)
                entries.resize(limit);
            return entries;
        }

        void maybeDumpPeriodicLocked()
        {
            const auto flushCalls = overrideTelemetryFlushCalls();
            if(flushCalls == 0 || m_totalCalls == 0)
                return;
            if(m_nextFlushAt == 0)
                m_nextFlushAt = flushCalls;
            if(m_totalCalls < m_nextFlushAt)
                return;

            dumpSummaryLocked("periodic");
            m_nextFlushAt += flushCalls;
        }

        void dumpSummary(const char* reason)
        {
            if(!overrideTelemetryEnabled())
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            dumpSummaryLocked(reason);
        }

        void dumpSummaryLocked(const char* reason)
        {
            if(m_totalCalls == 0)
                return;

            std::fprintf(stderr,
                         "[hipblaslt][override-summary] rank=%s local_rank=%s reason=%s calls=%llu "
                         "c_calls=%llu cpp_calls=%llu cache_hits=%llu cache_misses=%llu "
                         "success=%llu failure=%llu exact=%llu arch_default=%llu legacy=%llu "
                         "no_match=%llu candidate_attempts=%llu candidate_rejects=%llu "
                         "unique_problem_keys=%zu unique_cache_keys=%zu\n",
                         overrideTelemetryRank(),
                         overrideTelemetryLocalRank(),
                         reason,
                         static_cast<unsigned long long>(m_totalCalls),
                         static_cast<unsigned long long>(m_cApiCalls),
                         static_cast<unsigned long long>(m_cppApiCalls),
                         static_cast<unsigned long long>(m_cacheHits),
                         static_cast<unsigned long long>(m_cacheMisses),
                         static_cast<unsigned long long>(m_successes),
                         static_cast<unsigned long long>(m_failures),
                         static_cast<unsigned long long>(m_exactLookups),
                         static_cast<unsigned long long>(m_archDefaultLookups),
                         static_cast<unsigned long long>(m_legacyLookups),
                         static_cast<unsigned long long>(m_noMatchLookups),
                         static_cast<unsigned long long>(m_candidateAttempts),
                         static_cast<unsigned long long>(m_candidateRejects),
                         m_uniqueProblemKeys.size(),
                         m_uniqueCacheKeys.size());

            const auto callsAsDouble = static_cast<double>(m_totalCalls);
            if(callsAsDouble > 0.0)
            {
                const auto avgTotalUs
                    = static_cast<double>(m_totalNs) / (1000.0 * callsAsDouble);
                const auto avgPreloadUs
                    = static_cast<double>(m_preloadNs) / (1000.0 * callsAsDouble);
                const auto avgCacheProbeUs
                    = static_cast<double>(m_cacheProbeNs) / (1000.0 * callsAsDouble);
                const auto avgLookupUs
                    = static_cast<double>(m_lookupNs) / (1000.0 * callsAsDouble);
                const auto avgOtherUs = avgTotalUs - avgPreloadUs - avgCacheProbeUs - avgLookupUs;
                std::fprintf(stderr,
                             "[hipblaslt][override-summary] rank=%s local_rank=%s reason=%s avg_total_us=%.3f "
                             "avg_preload_us=%.3f avg_cache_probe_us=%.3f avg_lookup_us=%.3f "
                             "avg_other_us=%.3f\n",
                             overrideTelemetryRank(),
                             overrideTelemetryLocalRank(),
                             reason,
                             avgTotalUs,
                             avgPreloadUs,
                             avgCacheProbeUs,
                             avgLookupUs,
                             avgOtherUs);
            }

            std::fprintf(stderr,
                         "[hipblaslt][override-summary] rank=%s local_rank=%s reason=%s runtime_hint_controller_hit=%llu "
                         "runtime_hint_other_hit=%llu runtime_hint_no_snapshot=%llu runtime304_fallbacks=%llu "
                         "fallback_304_selected=%llu exact_272_selected=%llu\n",
                         overrideTelemetryRank(),
                         overrideTelemetryLocalRank(),
                         reason,
                         static_cast<unsigned long long>(m_runtimeControllerHits),
                         static_cast<unsigned long long>(m_runtimeNonControllerHits),
                         static_cast<unsigned long long>(m_runtimeNoSnapshot),
                         static_cast<unsigned long long>(m_runtime304Fallbacks),
                         static_cast<unsigned long long>(m_fallback304Selected),
                         static_cast<unsigned long long>(m_exact272Selected));

            for(const auto& [solutionIndex, count] : sortCountsDescending(m_selectedSolutionCounts, 12))
            {
                std::fprintf(stderr,
                             "[hipblaslt][override-summary] rank=%s local_rank=%s selected_solution index=%d count=%llu\n",
                             overrideTelemetryRank(),
                             overrideTelemetryLocalRank(),
                             solutionIndex,
                             static_cast<unsigned long long>(count));
            }

            for(const auto& [solutionIndex, count] : sortCountsDescending(m_rejectedSolutionCounts, 12))
            {
                std::fprintf(stderr,
                             "[hipblaslt][override-summary] rank=%s local_rank=%s rejected_solution index=%d count=%llu\n",
                             overrideTelemetryRank(),
                             overrideTelemetryLocalRank(),
                             solutionIndex,
                             static_cast<unsigned long long>(count));
            }

            std::vector<std::pair<std::string, KeyStats>> keyEntries(m_keyStats.begin(), m_keyStats.end());
            std::sort(keyEntries.begin(),
                      keyEntries.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.second.calls > rhs.second.calls; });
            if(keyEntries.size() > 12)
                keyEntries.resize(12);

            for(const auto& [key, stats] : keyEntries)
            {
                std::fprintf(stderr,
                             "[hipblaslt][override-summary] rank=%s local_rank=%s key=%s calls=%llu cache_hits=%llu "
                             "cache_misses=%llu success=%llu failure=%llu exact=%llu "
                             "arch_default=%llu legacy=%llu no_match=%llu candidate_attempts=%llu "
                             "candidate_rejects=%llu\n",
                             overrideTelemetryRank(),
                             overrideTelemetryLocalRank(),
                             key.c_str(),
                             static_cast<unsigned long long>(stats.calls),
                             static_cast<unsigned long long>(stats.cacheHits),
                             static_cast<unsigned long long>(stats.cacheMisses),
                             static_cast<unsigned long long>(stats.successes),
                             static_cast<unsigned long long>(stats.failures),
                             static_cast<unsigned long long>(stats.exactLookups),
                             static_cast<unsigned long long>(stats.archDefaultLookups),
                             static_cast<unsigned long long>(stats.legacyLookups),
                             static_cast<unsigned long long>(stats.noMatchLookups),
                             static_cast<unsigned long long>(stats.candidateAttempts),
                             static_cast<unsigned long long>(stats.candidateRejects));
            }
        }

        std::mutex                     m_mutex;
        uint64_t                       m_totalCalls         = 0;
        uint64_t                       m_cApiCalls          = 0;
        uint64_t                       m_cppApiCalls        = 0;
        uint64_t                       m_cacheHits          = 0;
        uint64_t                       m_cacheMisses        = 0;
        uint64_t                       m_successes          = 0;
        uint64_t                       m_failures           = 0;
        uint64_t                       m_exactLookups       = 0;
        uint64_t                       m_archDefaultLookups = 0;
        uint64_t                       m_legacyLookups      = 0;
        uint64_t                       m_noMatchLookups     = 0;
        uint64_t                       m_candidateAttempts  = 0;
        uint64_t                       m_candidateRejects   = 0;
        uint64_t                       m_totalNs            = 0;
        uint64_t                       m_preloadNs          = 0;
        uint64_t                       m_cacheProbeNs       = 0;
        uint64_t                       m_lookupNs           = 0;
        uint64_t                       m_nextFlushAt        = 0;
        uint64_t                       m_runtimeControllerHits   = 0;
        uint64_t                       m_runtimeNonControllerHits = 0;
        uint64_t                       m_runtimeNoSnapshot       = 0;
        uint64_t                       m_runtime304Fallbacks     = 0;
        uint64_t                       m_fallback304Selected     = 0;
        uint64_t                       m_exact272Selected        = 0;
        std::set<std::string>          m_uniqueProblemKeys;
        std::set<std::string>          m_uniqueCacheKeys;
        std::map<int, uint64_t>        m_selectedSolutionCounts;
        std::map<int, uint64_t>        m_rejectedSolutionCounts;
        std::map<std::string, KeyStats> m_keyStats;
    };

    inline void recordRuntimeHintTelemetry(bool success, bool controllerSource)
    {
        OverrideTelemetry::instance().recordRuntimeHintOutcome(success, controllerSource);
    }

    inline OverrideLookupOutcome
        lookupOverrideSolutions(TensileLite::OverrideMap&          overrideMap,
                                const TensileLite::ProblemOverride& baseKey,
                                const std::string&                 archName,
                                uint32_t                           cuCount)
    {
        OverrideLookupOutcome outcome;
        auto exactSolutions    = overrideMap.lookup(makeOverrideLookupKey(baseKey, archName, cuCount));
        auto archDefault       = overrideMap.lookup(makeOverrideLookupKey(baseKey, archName, 0));
        auto legacySolutions   = overrideMap.lookup(makeOverrideLookupKey(baseKey));
        outcome.exactCount       = exactSolutions.size();
        outcome.archDefaultCount = archDefault.size();
        outcome.legacyCount      = legacySolutions.size();

#ifdef HIPBLASLT_ENABLE_UOPC
        if(uopcDebugEnabled())
        {
            std::fprintf(stderr,
                         "[hipblaslt][uopc] override lookup arch=%s cu=%u m=%zu n=%zu k=%zu b=%zu "
                         "act=%s bias=%d exact=%zu arch_default=%zu legacy=%zu\n",
                         archName.c_str(),
                         cuCount,
                         baseKey.m(),
                         baseKey.n(),
                         baseKey.k(),
                         baseKey.batchSize(),
                         baseKey.activationType().c_str(),
                         baseKey.biasVector(),
                         outcome.exactCount,
                         outcome.archDefaultCount,
                         outcome.legacyCount);
        }
#endif

        if(!exactSolutions.empty())
        {
            outcome.tier            = OverrideLookupTier::exact;
            outcome.candidates      = std::move(exactSolutions);
            outcome.resolvedCuCount = cuCount;
            return outcome;
        }
        if(!archDefault.empty())
        {
            outcome.tier            = OverrideLookupTier::arch_default;
            outcome.candidates      = std::move(archDefault);
            outcome.resolvedCuCount = 0;
            return outcome;
        }
        if(!legacySolutions.empty())
        {
            outcome.tier            = OverrideLookupTier::legacy;
            outcome.candidates      = std::move(legacySolutions);
            outcome.resolvedCuCount = 0;
            return outcome;
        }
        return outcome;
    }

    inline OverrideLookupOutcome
        lookupScopedExactOverrideSolutions(TensileLite::OverrideMap&          overrideMap,
                                           const TensileLite::ProblemOverride& baseKey,
                                           const std::string&                 archName,
                                           uint32_t                           cuCount)
    {
        OverrideLookupOutcome outcome;
        auto                  exactSolutions
            = overrideMap.lookup(makeOverrideLookupKey(baseKey, archName, cuCount));
        outcome.exactCount = exactSolutions.size();

#ifdef HIPBLASLT_ENABLE_UOPC
        if(uopcDebugEnabled())
        {
            std::fprintf(stderr,
                         "[hipblaslt][uopc] scoped exact lookup arch=%s cu=%u m=%zu n=%zu k=%zu b=%zu "
                         "act=%s bias=%d exact=%zu\n",
                         archName.c_str(),
                         cuCount,
                         baseKey.m(),
                         baseKey.n(),
                         baseKey.k(),
                         baseKey.batchSize(),
                         baseKey.activationType().c_str(),
                         baseKey.biasVector(),
                         outcome.exactCount);
        }
#endif

        if(!exactSolutions.empty())
        {
            outcome.tier            = OverrideLookupTier::exact;
            outcome.candidates      = std::move(exactSolutions);
            outcome.resolvedCuCount = cuCount;
        }
        return outcome;
    }
} // namespace

inline void assignAlphaBeta1(const rocblaslt_compute_type& compute_type, void* alpha, void* beta)
{
    if(compute_type == rocblaslt_compute_f64)
    {
        *((double*)alpha) = 1.f;
        *((double*)beta)  = 1.f;
    }
    else if(compute_type == rocblaslt_compute_i32)
    {
        *((int32_t*)alpha) = 1.f;
        *((int32_t*)beta)  = 1.f;
    }
    else if(compute_type == rocblaslt_compute_f16)
    {
        *((hipblasLtHalf*)alpha) = 1.f;
        *((hipblasLtHalf*)beta)  = 1.f;
    }
    else
    {
        *((float*)alpha) = 1.f;
        *((float*)beta)  = 1.f;
    }
}

inline void heuristicResult_copy(rocblaslt_matmul_heuristic_result* heuristicResultsDest,
                                 rocblaslt_matmul_heuristic_result* heuristicResultsSrc,
                                 size_t&                            maxWorkSpaceBytes,
                                 size_t&                            required_workspace_size)
{
    memcpy(heuristicResultsDest->algo.data,
           heuristicResultsSrc->algo.data,
           sizeof(heuristicResultsDest->algo.data));
    heuristicResultsDest->algo.max_workspace_bytes = maxWorkSpaceBytes;
    heuristicResultsDest->algo.fallback            = false;
    heuristicResultsDest->state                    = rocblaslt_status_success;
    heuristicResultsDest->workspaceSize            = required_workspace_size;
}

inline bool
    heuristicResult_check_duplicated(rocblaslt_matmul_heuristic_result* heuristicResultsArray,
                                     rocblaslt_matmul_heuristic_result* SolutionsResult,
                                     int&                               AlgoCount,
                                     bool                               override_option)
{

    int index = -1;

    for(int i = 0; i < AlgoCount; i++)
    {
        if(*(int*)(heuristicResultsArray[i].algo.data)
           == *(int*)(SolutionsResult->algo.data)) //solution index
            index = i;
    }

    if(override_option && index != -1)
    {
        for(int i = index; i < AlgoCount - 1; i++)
        {
            heuristicResultsArray[i] = heuristicResultsArray[i + 1];
        }
    }

    return (index == -1) ? false : true;
}

bool problem_scoped_override_from_file(rocblaslt_handle&                 handle,
                                       RocblasltContractionProblem&      problem,
                                       rocblaslt_matmul_desc&            matmul_desc,
                                       rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                       const std::string&                file_path,
                                       size_t                            max_workspace_bytes)
{
    using Clock = std::chrono::steady_clock;
    const auto callStart = Clock::now();
    if(!shouldUseScopedUopcExact(problem))
        return false;

    bool success = false;
    const auto preloadStart = Clock::now();
    TensileLite::getContractionProblemsFromFile(file_path);
    const auto preloadEnd = Clock::now();
    TensileLite::OverrideMap& m_override = TensileLite::OverrideMap::getMap();

    if(m_override.size() == 0)
    {
        log_info(__func__, "No valid entries found in scoped override file.");
        return false;
    }

    std::vector<rocblaslt_matmul_heuristic_result> overrideResults;
    std::vector<int>                               solutionIndex(1);
    const auto                                     baseKey
        = RocblasltContractionProblem2ProblemOverride(problem);
    const auto archName          = getOverrideArchName(handle);
    const auto requestedCuCount  = getOverrideCuCount(handle, problem);
    const auto visibleCuCount    = getVisibleCuCount(handle);
    const auto hintSource        = static_cast<uint32_t>(problem.uopc.source);
    const auto overlapWindowType = static_cast<uint32_t>(problem.uopc.overlapWindowType);
    const auto relatedSeqNumber  = static_cast<uint64_t>(problem.uopc.relatedSeqNumber);
    if(requestedCuCount == 0)
        return false;

    const auto requestedKey = makeOverrideLookupKey(baseKey, archName, requestedCuCount);
    const auto cacheKey     = makeScopedOverrideCacheKey(requestedKey, max_workspace_bytes);
    CachedOverrideResult cachedResult;
    const auto         cacheProbeStart = Clock::now();
    const bool cacheHit = OverrideHeuristicCache::instance().lookup(cacheKey, cachedResult);
    const auto cacheProbeEnd = Clock::now();
    OverrideTelemetry::instance().recordCacheLookup(
        OverrideApiPath::c_api, requestedKey, cacheKey, cacheHit);
    if(cacheHit)
    {
        OverrideTelemetry::instance().recordLookupTier(requestedKey, cachedResult.tier);
        OverrideTelemetry::instance().recordFinalResult(
            requestedKey,
            cachedResult.tier,
            cachedResult.success,
            cachedResult.success ? cachedResult.solutionIndex : -1);
        OverrideTelemetry::instance().recordDurations(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                     - cacheProbeStart)
                    .count()),
            0);
        if(cachedResult.success)
        {
            heuristicResult_copy(&heuristicResultsArray[0],
                                 &cachedResult.result,
                                 max_workspace_bytes,
                                 cachedResult.result.workspaceSize);
            if(cachedResult.resolvedCuCount > 0)
                setOverrideAlgoStoredEffectiveCuCount(heuristicResultsArray[0].algo,
                                                      cachedResult.resolvedCuCount);
        }
        dumpOverrideTrace(OverrideApiPath::c_api,
                          cachedResult.success ? "scoped_exact_cache_success"
                                               : "scoped_exact_cache_failure",
                          requestedKey,
                          cachedResult.tier,
                          requestedCuCount,
                          cachedResult.resolvedCuCount > 0 ? cachedResult.resolvedCuCount
                                                           : requestedCuCount,
                          visibleCuCount,
                          true,
                          cachedResult.success,
                          cachedResult.success ? cachedResult.solutionIndex : -1,
                          hintSource,
                          overlapWindowType,
                          relatedSeqNumber);
        return cachedResult.success;
    }

    const auto lookupStart   = Clock::now();
    const auto lookupOutcome = lookupScopedExactOverrideSolutions(
        m_override, baseKey, archName, requestedCuCount);
    const auto lookupEnd = Clock::now();
    const auto telemetryKey = makeOverrideLookupKey(baseKey, archName, requestedCuCount);
    OverrideTelemetry::instance().recordLookupTier(telemetryKey, lookupOutcome.tier);

    for(auto candidateIdx : lookupOutcome.candidates)
    {
        solutionIndex[0] = candidateIdx;
        overrideResults.clear();
        bool candidateAccepted = false;

        if(rocblaslt_status_success
           == getSolutionsFromIndex(
               handle, solutionIndex, overrideResults, max_workspace_bytes))
        {
            size_t required_workspace_size = 0;
            auto&  tensile_data            = matmul_desc->m_data;

            if(rocblaslt_status_success
               == isSolutionSupported(handle,
                                      problem,
                                      tensile_data,
                                      &overrideResults[0].algo,
                                      &required_workspace_size))
            {
                success = true;
            }
            else if(problem.compute_type == rocblaslt_compute_f32_fast_xf32)
            {
                problem.compute_type = rocblaslt_compute_f32;
                if(rocblaslt_status_success
                   == isSolutionSupported(handle,
                                          problem,
                                          tensile_data,
                                          &overrideResults[0].algo,
                                          &required_workspace_size))
                {
                    success = true;
                    log_info(__func__, "Use the fallback fp32 solution");
                }
                problem.compute_type = rocblaslt_compute_f32_fast_xf32;
            }

            if(success)
            {
                candidateAccepted = true;
                heuristicResult_copy(&heuristicResultsArray[0],
                                     &overrideResults[0],
                                     max_workspace_bytes,
                                     required_workspace_size);
                setOverrideAlgoStoredEffectiveCuCount(
                    heuristicResultsArray[0].algo, requestedCuCount);
                CachedOverrideResult cachedSuccess;
                cachedSuccess.initialized     = true;
                cachedSuccess.success         = true;
                cachedSuccess.solutionIndex   = solutionIndex[0];
                cachedSuccess.resolvedCuCount = requestedCuCount;
                cachedSuccess.tier            = lookupOutcome.tier;
                cachedSuccess.result          = heuristicResultsArray[0];
                OverrideHeuristicCache::instance().store(cacheKey, cachedSuccess);
            }
        }

        OverrideTelemetry::instance().recordCandidateAttempt(
            telemetryKey, solutionIndex[0], candidateAccepted);
        if(candidateAccepted)
            break;
    }

    if(!success)
    {
        CachedOverrideResult cachedFailure;
        cachedFailure.initialized     = true;
        cachedFailure.success         = false;
        cachedFailure.resolvedCuCount = requestedCuCount;
        cachedFailure.tier            = lookupOutcome.tier;
        OverrideHeuristicCache::instance().store(cacheKey, cachedFailure);
    }
    else
    {
        std::string mapping_result = "Find scoped exact solution with index: ";
        mapping_result += std::to_string(solutionIndex[0]);
        log_info(__func__, mapping_result);
    }

    dumpOverrideTrace(OverrideApiPath::c_api,
                      success ? "scoped_exact_selected" : "scoped_exact_no_match",
                      telemetryKey,
                      lookupOutcome.tier,
                      requestedCuCount,
                      requestedCuCount,
                      visibleCuCount,
                      false,
                      success,
                      success ? solutionIndex[0] : -1,
                      hintSource,
                      overlapWindowType,
                      relatedSeqNumber);
    OverrideTelemetry::instance().recordFinalResult(
        telemetryKey, lookupOutcome.tier, success, success ? solutionIndex[0] : -1);
    OverrideTelemetry::instance().recordDurations(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd - cacheProbeStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(lookupEnd - lookupStart)
                .count()));
    return success;
}

bool problem_scoped_override_from_file_cpp(
    rocblaslt_handle&                               handle,
    rocblaslt::RocGemmType&                         gemmType,
    std::shared_ptr<void>                           gemmData,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResultsArray,
    const std::string&                              file_path,
    size_t                                          max_workspace_bytes)
{
    using Clock = std::chrono::steady_clock;
    const auto callStart = Clock::now();
    if(gemmType != rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        return false;

    const auto overrideCuCount = TensileDataGemmEffectiveCuCount(gemmData);
    if(overrideCuCount == 0)
        return false;

    bool success = false;
    const auto preloadStart = Clock::now();
    TensileLite::getContractionProblemsFromFile(file_path);
    const auto preloadEnd = Clock::now();
    TensileLite::OverrideMap& m_override = TensileLite::OverrideMap::getMap();
    if(m_override.size() == 0)
    {
        log_info(__func__, "No valid entries found in scoped override file.");
        return false;
    }

    std::vector<rocblaslt_matmul_heuristic_result> overrideResults;
    std::vector<int>                               solutionIndex(1);
    const auto resolvedCuCount = overrideCuCount;
    const auto baseKey
        = TensileDataGemm2ProblemOverride(gemmData, getOverrideArchName(handle), resolvedCuCount);
    const auto cacheKey = makeScopedOverrideCacheKey(baseKey, max_workspace_bytes);
    CachedOverrideResult cachedResult;
    const auto         cacheProbeStart = Clock::now();
    const bool cacheHit = OverrideHeuristicCache::instance().lookup(cacheKey, cachedResult);
    const auto cacheProbeEnd = Clock::now();
    OverrideTelemetry::instance().recordCacheLookup(
        OverrideApiPath::cpp_api, baseKey, cacheKey, cacheHit);
    if(cacheHit)
    {
        OverrideTelemetry::instance().recordLookupTier(baseKey, cachedResult.tier);
        OverrideTelemetry::instance().recordFinalResult(
            baseKey,
            cachedResult.tier,
            cachedResult.success,
            cachedResult.success ? cachedResult.solutionIndex : -1);
        OverrideTelemetry::instance().recordDurations(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                     - cacheProbeStart)
                    .count()),
            0);
        if(cachedResult.success)
            heuristicResultsArray.push_back(cachedResult.result);
        dumpOverrideTrace(OverrideApiPath::cpp_api,
                          cachedResult.success ? "scoped_exact_cache_success"
                                               : "scoped_exact_cache_failure",
                          baseKey,
                          cachedResult.tier,
                          resolvedCuCount,
                          cachedResult.resolvedCuCount > 0 ? cachedResult.resolvedCuCount
                                                           : resolvedCuCount,
                          0U,
                          true,
                          cachedResult.success,
                          cachedResult.success ? cachedResult.solutionIndex : -1,
                          0U,
                          0U,
                          0ULL);
        return cachedResult.success;
    }

    const auto lookupStart   = Clock::now();
    const auto lookupOutcome = lookupScopedExactOverrideSolutions(
        m_override, baseKey, getOverrideArchName(handle), resolvedCuCount);
    const auto lookupEnd = Clock::now();
    OverrideTelemetry::instance().recordLookupTier(baseKey, lookupOutcome.tier);

    for(auto candidateIdx : lookupOutcome.candidates)
    {
        solutionIndex[0] = candidateIdx;
        overrideResults.clear();
        bool candidateAccepted = false;
        if(rocblaslt_status_success
           == getSolutionsFromIndex(
               handle, solutionIndex, overrideResults, max_workspace_bytes))
        {
            size_t                  required_workspace_size = 0;
            rocblaslt::RocTuningV2* tuning                  = nullptr;
            if(rocblaslt_status_success
               == isSolutionSupported(handle,
                                      static_cast<const rocblaslt::RocGemmType>(gemmType),
                                      gemmData,
                                      overrideResults[0].algo,
                                      tuning,
                                      required_workspace_size))
            {
                success = true;
            }
            else
            {
                auto problem = ExtractProblemGemm(gemmData);
                if(problem->f32XdlMathOp() == rocisa::DataType::XFloat32)
                {
                    problem->setF32XdlMathOp(rocisa::DataType::Float);
                    if(rocblaslt_status_success
                       == isSolutionSupported(handle,
                                              static_cast<const rocblaslt::RocGemmType>(gemmType),
                                              gemmData,
                                              overrideResults[0].algo,
                                              tuning,
                                              required_workspace_size))
                    {
                        success = true;
                        log_info(__func__, "Use the fallback fp32 solution");
                    }
                }
            }

            if(success)
            {
                candidateAccepted = true;
                overrideResults[0].workspaceSize = required_workspace_size;
                heuristicResultsArray.push_back(overrideResults[0]);
                CachedOverrideResult cachedSuccess;
                cachedSuccess.initialized     = true;
                cachedSuccess.success         = true;
                cachedSuccess.solutionIndex   = solutionIndex[0];
                cachedSuccess.resolvedCuCount = resolvedCuCount;
                cachedSuccess.tier            = lookupOutcome.tier;
                cachedSuccess.result          = overrideResults[0];
                OverrideHeuristicCache::instance().store(cacheKey, cachedSuccess);
            }
        }

        OverrideTelemetry::instance().recordCandidateAttempt(
            baseKey, solutionIndex[0], candidateAccepted);
        if(candidateAccepted)
            break;
    }

    if(!success)
    {
        CachedOverrideResult cachedFailure;
        cachedFailure.initialized     = true;
        cachedFailure.success         = false;
        cachedFailure.resolvedCuCount = resolvedCuCount;
        cachedFailure.tier            = lookupOutcome.tier;
        OverrideHeuristicCache::instance().store(cacheKey, cachedFailure);
    }
    else
    {
        std::string mapping_result = "Find scoped exact solution with index: ";
        mapping_result += std::to_string(solutionIndex[0]);
        log_info(__func__, mapping_result);
    }

    dumpOverrideTrace(OverrideApiPath::cpp_api,
                      success ? "scoped_exact_selected" : "scoped_exact_no_match",
                      baseKey,
                      lookupOutcome.tier,
                      resolvedCuCount,
                      resolvedCuCount,
                      0U,
                      false,
                      success,
                      success ? solutionIndex[0] : -1,
                      0U,
                      0U,
                      0ULL);
    OverrideTelemetry::instance().recordFinalResult(
        baseKey, lookupOutcome.tier, success, success ? solutionIndex[0] : -1);
    OverrideTelemetry::instance().recordDurations(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd - cacheProbeStart)
                .count()),
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(lookupEnd - lookupStart)
                .count()));
    return success;
}

// Preload problem/solution mappings
bool problem_override_from_file(rocblaslt_handle&                 handle,
                                RocblasltContractionProblem&      problem,
                                rocblaslt_matmul_desc&            matmul_desc,
                                rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                const std::string&                file_path,
                                size_t                            max_workspace_bytes)
{
    using Clock = std::chrono::steady_clock;
    const auto callStart = Clock::now();
    bool success = false;
    const auto preloadStart = Clock::now();
    TensileLite::getContractionProblemsFromFile(file_path);
    const auto preloadEnd = Clock::now();
    TensileLite::OverrideMap& m_override = TensileLite::OverrideMap::getMap();

    if(m_override.size() == 0)
    {
        log_info(__func__, "No valid entries found in override file.");
    }
    else
    {
        std::vector<rocblaslt_matmul_heuristic_result> overrideResults;
        std::vector<int>                               solutionIndex(1);
        const auto                                     baseKey
            = RocblasltContractionProblem2ProblemOverride(problem);
        const auto archName          = getOverrideArchName(handle);
        const auto requestedCuCount  = getOverrideCuCount(handle, problem);
        const auto visibleCuCount    = getVisibleCuCount(handle);
        const auto hintSource
            = problem.uopc.valid ? static_cast<uint32_t>(problem.uopc.source) : 0U;
        const auto overlapWindowType
            = problem.uopc.valid ? static_cast<uint32_t>(problem.uopc.overlapWindowType) : 0U;
        const auto relatedSeqNumber
            = problem.uopc.valid ? static_cast<uint64_t>(problem.uopc.relatedSeqNumber) : 0ULL;
        const bool allow304Fallback
            = problem.uopc.valid && problem.uopc.source != kUopcHintSourcePreference
              && requestedCuCount > 0 && visibleCuCount > 0 && requestedCuCount != visibleCuCount;
        const auto requestedKey
            = makeOverrideLookupKey(baseKey, archName, requestedCuCount);
        const auto cacheKey = makeOverrideCacheKey(requestedKey, max_workspace_bytes);
        CachedOverrideResult cachedResult;
        const auto         cacheProbeStart = Clock::now();
        const bool cacheHit = OverrideHeuristicCache::instance().lookup(cacheKey, cachedResult);
        const auto cacheProbeEnd = Clock::now();
        OverrideTelemetry::instance().recordCacheLookup(
            OverrideApiPath::c_api, requestedKey, cacheKey, cacheHit);
        if(cacheHit)
        {
            const auto cachedCuCount
                = cachedResult.resolvedCuCount > 0 ? cachedResult.resolvedCuCount : requestedCuCount;
            const auto cachedKey = makeOverrideLookupKey(baseKey, archName, cachedCuCount);
            OverrideTelemetry::instance().recordLookupTier(cachedKey, cachedResult.tier);
            OverrideTelemetry::instance().recordFinalResult(
                cachedKey,
                cachedResult.tier,
                cachedResult.success,
                cachedResult.success ? cachedResult.solutionIndex : -1);
            OverrideTelemetry::instance().recordDurations(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                        .count()),
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                        .count()),
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                         - cacheProbeStart)
                        .count()),
                0);
            if(cachedResult.success)
            {
                heuristicResult_copy(&heuristicResultsArray[0],
                                     &cachedResult.result,
                                     max_workspace_bytes,
                                     cachedResult.result.workspaceSize);
                if(cachedResult.resolvedCuCount > 0)
                    setOverrideAlgoStoredEffectiveCuCount(heuristicResultsArray[0].algo,
                                                          cachedResult.resolvedCuCount);
            }
            else if(allow304Fallback)
            {
                clearProblemUopcForRuntimeFallback(problem);
                OverrideTelemetry::instance().recordRuntime304Fallback();
            }
            dumpOverrideTrace(OverrideApiPath::c_api,
                              cachedResult.success ? "cache_success"
                                                   : (allow304Fallback ? "runtime304_fallback_cache"
                                                                       : "cache_failure"),
                              cachedKey,
                              cachedResult.tier,
                              requestedCuCount,
                              cachedResult.success
                                  ? cachedCuCount
                                  : (allow304Fallback ? visibleCuCount : cachedCuCount),
                              visibleCuCount,
                              true,
                              cachedResult.success,
                              cachedResult.success ? cachedResult.solutionIndex : -1,
                              hintSource,
                              overlapWindowType,
                              relatedSeqNumber);
            return cachedResult.success;
        }
        const auto lookupStart = Clock::now();
        const auto lookupOutcome = lookupOverrideSolutions(
            m_override, baseKey, archName, requestedCuCount);
        const auto lookupEnd = Clock::now();
        const auto selectedCuCount
            = lookupOutcome.resolvedCuCount > 0 ? lookupOutcome.resolvedCuCount : requestedCuCount;
        const auto telemetryKey = makeOverrideLookupKey(baseKey, archName, selectedCuCount);
        OverrideTelemetry::instance().recordLookupTier(telemetryKey, lookupOutcome.tier);

        for(auto candidateIdx : lookupOutcome.candidates)
        {
            solutionIndex[0] = candidateIdx;
            overrideResults.clear();
            bool candidateAccepted = false;

            if(rocblaslt_status_success
               == getSolutionsFromIndex(
                   handle, solutionIndex, overrideResults, max_workspace_bytes))
            {

                size_t required_workspace_size = 0;
                auto&  tensile_data            = matmul_desc->m_data;

                if(rocblaslt_status_success
                   == isSolutionSupported(handle,
                                          problem,
                                          tensile_data,
                                          &overrideResults[0].algo,
                                          &required_workspace_size))
                {
                    success = true;
                }
                else
                { // there is no solution for xfloat32, fallback comput_type to fp32
                    if(problem.compute_type == rocblaslt_compute_f32_fast_xf32)
                    {
                        problem.compute_type = rocblaslt_compute_f32;
                        if(rocblaslt_status_success
                           == isSolutionSupported(handle,
                                                  problem,
                                                  tensile_data,
                                                  &overrideResults[0].algo,
                                                  &required_workspace_size))
                        {
                            success = true;
                            log_info(__func__, "Use the fallback fp32 solution");
                        }

                        problem.compute_type = rocblaslt_compute_f32_fast_xf32;
                    }
                }

                if(success)
                {
                    candidateAccepted = true;

                    heuristicResult_copy(&heuristicResultsArray[0],
                                         &overrideResults[0],
                                         max_workspace_bytes,
                                         required_workspace_size);
                    if(selectedCuCount > 0)
                        setOverrideAlgoStoredEffectiveCuCount(heuristicResultsArray[0].algo,
                                                              selectedCuCount);
                    CachedOverrideResult cachedSuccess;
                    cachedSuccess.initialized    = true;
                    cachedSuccess.success        = true;
                    cachedSuccess.solutionIndex  = solutionIndex[0];
                    cachedSuccess.resolvedCuCount = selectedCuCount;
                    cachedSuccess.tier           = lookupOutcome.tier;
                    cachedSuccess.result         = heuristicResultsArray[0];
                    OverrideHeuristicCache::instance().store(cacheKey, cachedSuccess);
                }
            }

            OverrideTelemetry::instance().recordCandidateAttempt(
                telemetryKey, solutionIndex[0], candidateAccepted);
            if(candidateAccepted)
                break;
        }

        if(!success)
        {
            CachedOverrideResult cachedFailure;
            cachedFailure.initialized    = true;
            cachedFailure.success        = false;
            cachedFailure.resolvedCuCount = selectedCuCount;
            cachedFailure.tier           = lookupOutcome.tier;
            OverrideHeuristicCache::instance().store(cacheKey, cachedFailure);
            log_info(__func__, "No valid solution index found in override file.");
            if(allow304Fallback)
            {
                clearProblemUopcForRuntimeFallback(problem);
                OverrideTelemetry::instance().recordRuntime304Fallback();
            }
        }
        else
        {
            std::string mapping_result = "Find solution with index: ";
            mapping_result += std::to_string(solutionIndex[0]);
            log_info(__func__, mapping_result);
        }
        dumpOverrideTrace(OverrideApiPath::c_api,
                          success ? "selected"
                                  : (allow304Fallback ? "runtime304_fallback_no_match"
                                                      : "no_valid_solution"),
                          telemetryKey,
                          lookupOutcome.tier,
                          requestedCuCount,
                          success ? selectedCuCount
                                  : (allow304Fallback ? visibleCuCount : selectedCuCount),
                          visibleCuCount,
                          false,
                          success,
                          success ? solutionIndex[0] : -1,
                          hintSource,
                          overlapWindowType,
                          relatedSeqNumber);
        OverrideTelemetry::instance().recordFinalResult(
            telemetryKey, lookupOutcome.tier, success, success ? solutionIndex[0] : -1);
        OverrideTelemetry::instance().recordDurations(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                     - cacheProbeStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(lookupEnd - lookupStart)
                    .count()));
    }

    return success;
}

bool problem_override_from_file_cpp(
    rocblaslt_handle&                               handle,
    rocblaslt::RocGemmType&                         gemmType,
    std::shared_ptr<void>                           gemmData,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResultsArray,
    const std::string&                              file_path,
    size_t                                          max_workspace_bytes)
{
    using Clock = std::chrono::steady_clock;
    const auto callStart = Clock::now();
    bool success = false;
    if(gemmType != rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        log_info(__func__, "Skip override file lookup for non-GEMM problem type.");
        return false;
    }

    const auto preloadStart = Clock::now();
    TensileLite::getContractionProblemsFromFile(file_path);
    const auto preloadEnd = Clock::now();
    TensileLite::OverrideMap& m_override = TensileLite::OverrideMap::getMap();

    if(m_override.size() == 0)
    {
        log_info(__func__, "No valid entries found in override file.");
    }
    else
    {
        std::vector<rocblaslt_matmul_heuristic_result> overrideResults;
        std::vector<int>                               solutionIndex(1);
        const auto overrideCuCount = TensileDataGemmEffectiveCuCount(gemmData);
        const auto resolvedCuCount = overrideCuCount > 0 ? overrideCuCount : getVisibleCuCount(handle);
        const auto baseKey
            = TensileDataGemm2ProblemOverride(gemmData, getOverrideArchName(handle), resolvedCuCount);
        const auto         cacheKey = makeOverrideCacheKey(baseKey, max_workspace_bytes);
        CachedOverrideResult cachedResult;
        const auto         cacheProbeStart = Clock::now();
        const bool cacheHit = OverrideHeuristicCache::instance().lookup(cacheKey, cachedResult);
        const auto cacheProbeEnd = Clock::now();
        OverrideTelemetry::instance().recordCacheLookup(
            OverrideApiPath::cpp_api, baseKey, cacheKey, cacheHit);
        if(cacheHit)
        {
            OverrideTelemetry::instance().recordLookupTier(baseKey, cachedResult.tier);
            OverrideTelemetry::instance().recordFinalResult(
                baseKey,
                cachedResult.tier,
                cachedResult.success,
                cachedResult.success ? cachedResult.solutionIndex : -1);
            OverrideTelemetry::instance().recordDurations(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                        .count()),
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                        .count()),
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                         - cacheProbeStart)
                        .count()),
                0);
            if(cachedResult.success)
            {
                heuristicResultsArray.push_back(cachedResult.result);
            }
            dumpOverrideTrace(OverrideApiPath::cpp_api,
                              cachedResult.success ? "cache_success" : "cache_failure",
                              baseKey,
                              cachedResult.tier,
                              resolvedCuCount,
                              cachedResult.resolvedCuCount > 0 ? cachedResult.resolvedCuCount
                                                               : resolvedCuCount,
                              0U,
                              true,
                              cachedResult.success,
                              cachedResult.success ? cachedResult.solutionIndex : -1,
                              0U,
                              0U,
                              0ULL);
            return cachedResult.success;
        }
        const auto lookupStart = Clock::now();
        const auto lookupOutcome = lookupOverrideSolutions(
            m_override, baseKey, getOverrideArchName(handle), resolvedCuCount);
        const auto lookupEnd = Clock::now();
        OverrideTelemetry::instance().recordLookupTier(baseKey, lookupOutcome.tier);

        for(auto candidateIdx : lookupOutcome.candidates)
        {
            solutionIndex[0] = candidateIdx;
            overrideResults.clear();
            bool candidateAccepted = false;
            if(rocblaslt_status_success
               == getSolutionsFromIndex(
                   handle, solutionIndex, overrideResults, max_workspace_bytes))
            {

                size_t                  required_workspace_size = 0;
                rocblaslt::RocTuningV2* tuning                  = nullptr;

                if(rocblaslt_status_success
                   == isSolutionSupported(handle,
                                          static_cast<const rocblaslt::RocGemmType>(gemmType),
                                          gemmData,
                                          overrideResults[0].algo,
                                          tuning,
                                          required_workspace_size))
                {
                    success = true;
                }
                else
                { // there is no solution for xfloat32, fallback comput_type to fp32
                    auto problem = ExtractProblemGemm(gemmData);
                    if(problem->f32XdlMathOp() == rocisa::DataType::XFloat32)
                    {
                        problem->setF32XdlMathOp(rocisa::DataType::Float);
                        if(rocblaslt_status_success
                           == isSolutionSupported(
                               handle,
                               static_cast<const rocblaslt::RocGemmType>(gemmType),
                               gemmData,
                               overrideResults[0].algo,
                               tuning,
                               required_workspace_size))
                        {
                            success = true;
                            log_info(__func__, "Use the fallback fp32 solution");
                        }
                    }
                }

                if(success)
                {
                    candidateAccepted = true;
                    overrideResults[0].workspaceSize = required_workspace_size;
                    heuristicResultsArray.push_back(overrideResults[0]);
                    CachedOverrideResult cachedSuccess;
                    cachedSuccess.initialized   = true;
                    cachedSuccess.success       = true;
                    cachedSuccess.solutionIndex = solutionIndex[0];
                    cachedSuccess.tier          = lookupOutcome.tier;
                    cachedSuccess.result        = overrideResults[0];
                    OverrideHeuristicCache::instance().store(cacheKey, cachedSuccess);
                }
            }

            OverrideTelemetry::instance().recordCandidateAttempt(
                baseKey, solutionIndex[0], candidateAccepted);
            if(candidateAccepted)
                break;
        }

        if(!success)
        {
            CachedOverrideResult cachedFailure;
            cachedFailure.initialized = true;
            cachedFailure.success     = false;
            cachedFailure.tier        = lookupOutcome.tier;
            OverrideHeuristicCache::instance().store(cacheKey, cachedFailure);
            log_info(__func__, "No valid solution index found in override file.");
        }
        else
        {
            std::string mapping_result = "Find solution with index: ";
            mapping_result += std::to_string(solutionIndex[0]);
            log_info(__func__, mapping_result);
        }
        dumpOverrideTrace(OverrideApiPath::cpp_api,
                          success ? "selected" : "no_valid_solution",
                          baseKey,
                          lookupOutcome.tier,
                          resolvedCuCount,
                          lookupOutcome.resolvedCuCount > 0 ? lookupOutcome.resolvedCuCount
                                                            : resolvedCuCount,
                          0U,
                          false,
                          success,
                          success ? solutionIndex[0] : -1,
                          0U,
                          0U,
                          0ULL);
        OverrideTelemetry::instance().recordFinalResult(
            baseKey, lookupOutcome.tier, success, success ? solutionIndex[0] : -1);
        OverrideTelemetry::instance().recordDurations(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - callStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(preloadEnd - preloadStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(cacheProbeEnd
                                                                     - cacheProbeStart)
                    .count()),
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(lookupEnd - lookupStart)
                    .count()));
    }

    return success;
}

/******************************************************************************
 * construct_rocblaslt_problem creates RocblasltContractionProblem from mat    *
 * layout and descriptor for Tensile's findTopSolutions.                      *
 ******************************************************************************/
RocblasltContractionProblem construct_rocblaslt_problem(rocblaslt_handle            handle,
                                                        const rocblaslt_matmul_desc matmul_descr,
                                                        rocblaslt_matrix_layout     matA,
                                                        rocblaslt_matrix_layout     matB,
                                                        rocblaslt_matrix_layout     matC,
                                                        rocblaslt_matrix_layout     matD,
                                                        const void*                 alpha,
                                                        const void*                 beta,
                                                        size_t maxWorkSpaceBytes)
{
    int8_t      dummy;
    const void* dummy_ptr = &dummy;
    int64_t     m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipDataType            bias_type;
    hipDataType            aux_type;
    hipDataType            a_type, b_type, c_type, d_type;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *e = nullptr;
    bool                   gradient = false;
    bool swizzleA = matA->order != HIPBLASLT_ORDER_COL && matA->order != HIPBLASLT_ORDER_ROW;
    bool swizzleB = matB->order != HIPBLASLT_ORDER_COL && matB->order != HIPBLASLT_ORDER_ROW;

    rocblaslt_status isValid = rocblaslt_matmul_valid_args(matmul_descr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           a_type,
                                                           lda,
                                                           batch_stride_a,
                                                           b_type,
                                                           ldb,
                                                           batch_stride_b,
                                                           c_type,
                                                           ldc,
                                                           batch_stride_c,
                                                           d_type,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           e,
                                                           aux_type,
                                                           gradient,
                                                           compute_type,
                                                           swizzleA,
                                                           swizzleB);
    if(isValid != rocblaslt_status_continue)
    {
        m = 0;
        n = 0;
        k = 0;
    }

    // Internal assign
    hipblasOperation_t opA           = matmul_descr->op_A;
    hipblasOperation_t opB           = matmul_descr->op_B;
    int                num_batches_a = matA->batch_count;
    rocblaslt_epilogue epilogue      = matmul_descr->epilogue;
    void*              scaleA        = matmul_descr->scaleA;
    void*              scaleB        = matmul_descr->scaleB;
    void*              scaleC        = matmul_descr->scaleC;
    void*              scaleD        = matmul_descr->scaleD;
    void*              scaleE        = matmul_descr->scaleE;
    void*              amaxD         = matmul_descr->amaxD;

    // Others
    constexpr bool strided_batch = true;
    constexpr bool grouped_gemm  = false;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(matmul_descr->compute_type, (void*)alpha_1, &alpha);
    }

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        a_type,
                                        nullptr,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        b_type,
                                        nullptr,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        c_type,
                                        nullptr,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        d_type,
                                        nullptr,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        e,
                                        nullptr,
                                        lde,
                                        batch_stride_e,
                                        num_batches_a,
                                        strided_batch,
                                        grouped_gemm,
                                        gradient,
                                        compute_type,
                                        matmul_descr->scale_type,
                                        bias,
                                        scaleA,
                                        scaleB,
                                        scaleC,
                                        scaleD,
                                        scaleE,
                                        scaleAlphaVec,
                                        matmul_descr->scaleAType,
                                        matmul_descr->scaleBType,
                                        matmul_descr->scaleABlockRowSize,
                                        matmul_descr->scaleABlockColSize,
                                        matmul_descr->scaleBBlockRowSize,
                                        matmul_descr->scaleBBlockColSize,
                                        bias_type,
                                        aux_type,
                                        epilogue,
                                        amaxD,
                                        nullptr,
                                        maxWorkSpaceBytes,
                                        matmul_descr->act0,
                                        matmul_descr->act1,
                                        nullptr,
                                        handle->Synchronizer,
                                        swizzleA,
                                        swizzleB};

    return problem;
}

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocblaslt_handle is a structure holding the rocblaslt library context.
 * It must be initialized using rocblaslt_create()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblaslt_destroy().
 *******************************************************************************/
rocblaslt_status rocblaslt_create(rocblaslt_handle* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        log_error(__func__, "invalid handle pointer", handle);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        *handle = nullptr;
        // Allocate
        try
        {
            *handle = new _rocblaslt_handle();
            log_api(__func__, "handle[out]", *handle);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocblaslt_status rocblaslt_destroy(const rocblaslt_handle handle)
{
    if(handle == nullptr)
    {
        log_error(__func__, "handle", handle);
        return rocblaslt_status_invalid_value;
    }
    log_api(__func__, "handle", handle);
// Destruct
#ifdef HIPBLASLT_USE_ROCROLLER
    if(handle->rocroller_handle)
        rocroller_destroy_handle(handle->rocroller_handle);
#endif
    try
    {
        delete handle;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief rocblaslt_matrix_layout is a structure holding the rocblaslt matrix
 * content. It must be initialized using rocblaslt_matrix_layout_create()
 * and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocblaslt_matrix_layout_destory().
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_create(rocblaslt_matrix_layout* matDescr,
                                                hipDataType              valueType,
                                                uint64_t                 rows,
                                                uint64_t                 cols,
                                                int64_t                  ld)
{
    // Check if matDescr is valid
    if(matDescr == nullptr)
    {
        log_error(__func__, "invalid matDescr pointer", matDescr);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        *matDescr = nullptr;
        // Allocate
        try
        {
            *matDescr         = new _rocblaslt_matrix_layout();
            (*matDescr)->m    = rows;
            (*matDescr)->n    = cols;
            (*matDescr)->ld   = ld;
            (*matDescr)->type = valueType;
            log_api(__func__,
                    "matLayout[out]",
                    matDescr,
                    "type",
                    hipDataType_to_string(valueType),
                    "rows",
                    rows,
                    "cols",
                    cols,
                    "ld",
                    ld);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_destory(const rocblaslt_matrix_layout matDescr)
{
    if(matDescr == nullptr)
    {
        log_error(__func__, "matDescr", matDescr);
        return rocblaslt_status_invalid_pointer;
    }
    log_api(__func__, "matLayout", matDescr);
    // Destruct
    try
    {
        delete matDescr;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *descriptor.
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_set_attribute(rocblaslt_matrix_layout           matLayout,
                                                       rocblaslt_matrix_layout_attribute attr,
                                                       const void*                       buf,
                                                       size_t sizeInBytes)
{
    // Check if matLayout is valid
    if(matLayout == nullptr)
    {
        log_error(__func__, "invalid matLayout pointer", matLayout);
        return rocblaslt_status_invalid_handle;
    }
    else if(buf == nullptr)
    {
        log_error(__func__, "invalid buf pointer", buf);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid buf size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            switch(attr)
            {
            case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matLayout->batch_count, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matLayout->batch_stride, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_TYPE:
                if(sizeof(uint32_t) <= sizeInBytes)
                    memcpy(&matLayout->type, buf, sizeof(uint32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_ORDER:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matLayout->order, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_ROWS:
                if(sizeof(uint64_t) <= sizeInBytes)
                    memcpy(&matLayout->m, buf, sizeof(uint64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_COLS:
                if(sizeof(uint64_t) <= sizeInBytes)
                    memcpy(&matLayout->n, buf, sizeof(uint64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_LD:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matLayout->ld, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            default:
                log_error(__func__, "invalid attribute", attr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matLayout",
                    matLayout,
                    "attr",
                    rocblaslt_matrix_layout_attributes_to_string(attr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData",
                    (void*)(intptr_t)(*(int32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief Get the value of the specified attribute belonging to matrix
 *descriptor such as number of batches and their stride.
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_get_attribute(rocblaslt_matrix_layout           matLayout,
                                                       rocblaslt_matrix_layout_attribute attr,
                                                       void*                             buf,
                                                       size_t  sizeInBytes,
                                                       size_t* sizeWritten)

{
    if(matLayout == nullptr)
    {
        log_error(__func__, "invalid matLayout pointer", matLayout);
        return rocblaslt_status_invalid_handle;
    }
    else if(sizeInBytes == 0 && sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: sizeWritten can't be nullptr if sizeInBytes is 0");
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes != 0 && buf == nullptr)
    {
        log_error(__func__, "invalid pointer: buf can't be nullptr if sizeInBytes isn't 0");
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(attr)
            {
            case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matLayout->batch_count, sizeof(int32_t));
                break;
            case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
                if(sizeWritten)
                    *sizeWritten = sizeof(int64_t);
                if(sizeInBytes < sizeof(int64_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matLayout->batch_stride, sizeof(int64_t));
                break;
            default:
                log_error(__func__, "invalid attribute", attr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matLayout",
                    matLayout,
                    "attr",
                    rocblaslt_matrix_layout_attributes_to_string(attr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData[out]",
                    (void*)(intptr_t)(*(int32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_create(rocblaslt_matmul_desc* matmulDesc,
                                              rocblaslt_compute_type computeType,
                                              hipDataType            scaleType)
{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        *matmulDesc = nullptr;
        // Allocate
        try
        {
            switch(computeType)
            {
            case rocblaslt_compute_f16:
            case rocblaslt_compute_f32:
            case rocblaslt_compute_f32_fast_xf32:
            case rocblaslt_compute_f64:
            case rocblaslt_compute_i32:
            case rocblaslt_compute_f32_fast_f16:
            case rocblaslt_compute_f32_fast_bf16:
            case rocblaslt_compute_f32_fast_f8_fnuz:
            case rocblaslt_compute_f32_fast_bf8_fnuz:
            case rocblaslt_compute_f32_fast_f8bf8_fnuz:
            case rocblaslt_compute_f32_fast_bf8f8_fnuz:
            case rocblaslt_compute_f32_fast_f8:
            case rocblaslt_compute_f32_fast_bf8:
            case rocblaslt_compute_f32_fast_f8bf8:
            case rocblaslt_compute_f32_fast_bf8f8:
                break;
            default:
                log_error(__func__, "invalid compute type", computeType);
                throw rocblaslt_status_invalid_value;
            }

            if(scaleType != HIP_R_32F && scaleType != HIP_R_64F && scaleType != HIP_R_32I)
            {
                log_error(__func__, "invalid scale type", scaleType);
                throw rocblaslt_status_invalid_value;
            }

            *matmulDesc = new _rocblaslt_matmul_desc();

            (*matmulDesc)->compute_type          = computeType;
            (*matmulDesc)->compute_type_original = computeType;
            (*matmulDesc)->scale_type            = scaleType;
            auto computeTypeInit                 = computeType == rocblaslt_compute_f32_fast_xf32
                                                       ? rocblaslt_compute_f32
                                                       : computeType;
            auto dataType                        = HIP_R_32F;
            if(computeTypeInit == rocblaslt_compute_f64)
                dataType = HIP_R_64F;
            else if(computeType == rocblaslt_compute_i32)
                dataType = HIP_R_32I;

            initTensileGemmData(nullptr,
                                rocblaslt::RocGemmType::ROCBLASLT_GEMM,
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_N,
                                dataType,
                                dataType,
                                dataType,
                                dataType,
                                computeTypeInit,
                                0,
                                (*matmulDesc)->m_data);

            log_api(__func__,
                    "matmulDesc[out]",
                    matmulDesc,
                    "computeType",
                    rocblaslt_compute_type_to_string(computeType),
                    "scaleType",
                    hipDataType_to_string(scaleType));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }

        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_destroy(const rocblaslt_matmul_desc matmulDesc)
{
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_pointer;
    }
    log_api(__func__, "matmulDesc", matmulDesc);

    // Destruct
    try
    {
        delete matmulDesc;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_compute_type _matmul_desc_determine_compute_type(rocblaslt_matmul_desc matmulDesc)
{
    if(matmulDesc->compute_type_original == rocblaslt_compute_f32)
    {
        auto tciA = matmulDesc->compute_input_typeA;
        auto tciB = matmulDesc->compute_input_typeB;
        if(tciA == tciB && tciA == HIP_R_16F)
            return rocblaslt_compute_f32_fast_f16;
        else if(tciA == tciB && tciA == HIP_R_16BF)
            return rocblaslt_compute_f32_fast_bf16;
        else if(tciA == tciB && tciA == HIP_R_8F_E4M3_FNUZ)
            return rocblaslt_compute_f32_fast_f8_fnuz;
        else if(tciA == tciB && tciA == HIP_R_8F_E5M2_FNUZ)
            return rocblaslt_compute_f32_fast_bf8_fnuz;
        else if(tciA == HIP_R_8F_E4M3_FNUZ && tciB == HIP_R_8F_E5M2_FNUZ)
            return rocblaslt_compute_f32_fast_f8bf8_fnuz;
        else if(tciA == HIP_R_8F_E5M2_FNUZ && tciB == HIP_R_8F_E4M3_FNUZ)
            return rocblaslt_compute_f32_fast_bf8f8_fnuz;
        else if(tciA == tciB && tciA == HIP_R_8F_E4M3)
            return rocblaslt_compute_f32_fast_f8;
        else if(tciA == tciB && tciA == HIP_R_8F_E5M2)
            return rocblaslt_compute_f32_fast_bf8;
        else if(tciA == HIP_R_8F_E4M3 && tciB == HIP_R_8F_E5M2)
            return rocblaslt_compute_f32_fast_f8bf8;
        else if(tciA == HIP_R_8F_E5M2 && tciB == HIP_R_8F_E4M3)
            return rocblaslt_compute_f32_fast_bf8f8;
    }
    return matmulDesc->compute_type_original;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *multiplication descriptor.
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_set_attribute(rocblaslt_matmul_desc            matmulDesc,
                                                     rocblaslt_matmul_desc_attributes matmulAttr,
                                                     const void*                      buf,
                                                     size_t                           sizeInBytes)
{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_handle;
    }
    else if(buf == nullptr)
    {
        log_error(__func__, "invalid buf pointer", buf);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid buf size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            switch(matmulAttr)
            {
            case ROCBLASLT_MATMUL_DESC_TRANSA:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->op_A, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_TRANSB:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->op_B, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->epilogue, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->bias, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER:
                if(matmulDesc->scaleAType == RocblasltContractionProblem::ScalingFormat::None)
                {
                    matmulDesc->scaleAType = RocblasltContractionProblem::ScalingFormat::Scalar;
                }
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleA, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleA buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_MODE:
                if(sizeof(hipblasLtMatmulMatrixScale_t) <= sizeInBytes)
                {
                    hipblasLtMatmulMatrixScale_t mode;
                    memcpy(&mode, buf, sizeof(hipblasLtMatmulMatrixScale_t));
                    switch(mode)
                    {
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
                        matmulDesc->scaleABlockRowSize = 32;
                        matmulDesc->scaleABlockColSize = 1;
                        matmulDesc->scaleAType = RocblasltContractionProblem::ScalingFormat::Block;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:
                        matmulDesc->scaleABlockRowSize = 1;
                        matmulDesc->scaleABlockColSize = 1;
                        matmulDesc->scaleAType = RocblasltContractionProblem::ScalingFormat::Scalar;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F:
                        matmulDesc->scaleABlockRowSize = 1;
                        matmulDesc->scaleABlockColSize = 1;
                        matmulDesc->scaleAType = RocblasltContractionProblem::ScalingFormat::Vector;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F:
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F:
                    default:
                        log_error(__func__, "invalid A scale mode: ", mode);
                        return rocblaslt_status_invalid_value;
                    }
                }
                else
                {
                    log_error(__func__, "invalid A scale mode buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER:
                if(matmulDesc->scaleBType == RocblasltContractionProblem::ScalingFormat::None)
                {
                    matmulDesc->scaleBType = RocblasltContractionProblem::ScalingFormat::Scalar;
                }
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleB, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleB buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_MODE:
                if(sizeof(hipblasLtMatmulMatrixScale_t) <= sizeInBytes)
                {
                    hipblasLtMatmulMatrixScale_t mode;
                    memcpy(&mode, buf, sizeof(hipblasLtMatmulMatrixScale_t));
                    switch(mode)
                    {
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0:
                        matmulDesc->scaleBBlockRowSize = 1;
                        matmulDesc->scaleBBlockColSize = 32;
                        matmulDesc->scaleBType = RocblasltContractionProblem::ScalingFormat::Block;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F:
                        matmulDesc->scaleBBlockRowSize = 1;
                        matmulDesc->scaleBBlockColSize = 1;
                        matmulDesc->scaleBType = RocblasltContractionProblem::ScalingFormat::Scalar;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F:
                        matmulDesc->scaleBBlockRowSize = 1;
                        matmulDesc->scaleBBlockColSize = 1;
                        matmulDesc->scaleBType = RocblasltContractionProblem::ScalingFormat::Vector;
                        break;
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3:
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F:
                    case HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F:
                    default:
                        log_error(__func__, "invalid B scale mode: ", mode);
                        return rocblaslt_status_invalid_value;
                    }
                }
                else
                {
                    log_error(__func__, "invalid B scale mode buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleC, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleC buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleD, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleD buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleE, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleAux buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_POINTER_MODE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->pointermode, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid pointermode buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->bias_type, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->e, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid e buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matmulDesc->lde, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid lde buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matmulDesc->stride_e, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid stride_e buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->amaxD, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid amax buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->aux_type, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT:
                if(sizeof(int32_t) <= sizeInBytes)
                {
                    memcpy(&matmulDesc->compute_input_typeA, buf, sizeof(int32_t));
                    matmulDesc->compute_type = _matmul_desc_determine_compute_type(matmulDesc);
                }
                else
                {
                    log_error(__func__, "invalid compute_input_typeA buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT:
                if(sizeof(int32_t) <= sizeInBytes)
                {
                    memcpy(&matmulDesc->compute_input_typeB, buf, sizeof(int32_t));
                    matmulDesc->compute_type = _matmul_desc_determine_compute_type(matmulDesc);
                }
                else
                {
                    log_error(__func__, "invalid compute_input_typeB buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT:
                if(sizeof(float) <= sizeInBytes)
                    memcpy(&matmulDesc->act0, buf, sizeof(float));
                else
                {
                    log_error(__func__, "invalid act arg0 buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT:
                if(sizeof(float) <= sizeInBytes)
                    memcpy(&matmulDesc->act1, buf, sizeof(float));
                else
                {
                    log_error(__func__, "invalid act arg1 buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            default:
                log_error(__func__, "invalid attribute", matmulAttr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matmulDesc",
                    matmulDesc,
                    "attr",
                    rocblaslt_matmul_desc_attributes_to_string(matmulAttr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData",
                    (void*)(uintptr_t)(*(uint32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *descriptor such as number of batches and their stride.
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_get_attribute(rocblaslt_matmul_desc            matmulDesc,
                                                     rocblaslt_matmul_desc_attributes matmulAttr,
                                                     void*                            buf,
                                                     size_t                           sizeInBytes,
                                                     size_t*                          sizeWritten)

{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_handle;
    }
    else if(sizeInBytes == 0 && sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: sizeWritten can't be nullptr if sizeInBytes is 0");
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes != 0 && buf == nullptr)
    {
        log_error(__func__, "invalid pointer: buf can't be nullptr if sizeInBytes isn't 0");
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(matmulAttr)
            {
            case ROCBLASLT_MATMUL_DESC_TRANSA:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_A, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_TRANSB:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_B, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->epilogue, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->bias, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->scaleA, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_MODE: //TODO: May need to handle default value too.
                if(sizeWritten)
                    *sizeWritten = sizeof(uint32_t);
                if(sizeInBytes < sizeof(uint32_t))
                {
                    log_error(__func__, "invalid scale block A scale mode size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                else
                {
                    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                    if(matmulDesc->scaleABlockRowSize == 32 && matmulDesc->scaleABlockColSize == 1
                       && matmulDesc->scaleAType
                              == RocblasltContractionProblem::ScalingFormat::Block)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
                    }
                    else if(matmulDesc->scaleAType
                            == RocblasltContractionProblem::ScalingFormat::Scalar)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                    }
                    else if(matmulDesc->scaleAType
                            == RocblasltContractionProblem::ScalingFormat::Vector)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                    }
                    else
                    {
                        log_error(__func__, "invalid A scale mode", mode);
                        return rocblaslt_status_invalid_value;
                    }
                    memcpy(buf, &mode, sizeof(uint32_t));
                }
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->scaleB, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_MODE: //TODO: May need to handle default value too.
                if(sizeWritten)
                    *sizeWritten = sizeof(uint32_t);
                if(sizeInBytes < sizeof(uint32_t))
                {
                    log_error(__func__, "invalid scale block B scale mode size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                else
                {
                    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                    if(matmulDesc->scaleBBlockRowSize == 1 && matmulDesc->scaleBBlockColSize == 32
                       && matmulDesc->scaleBType
                              == RocblasltContractionProblem::ScalingFormat::Block)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
                    }
                    else if(matmulDesc->scaleBType
                            == RocblasltContractionProblem::ScalingFormat::Scalar)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                    }
                    else if(matmulDesc->scaleBType
                            == RocblasltContractionProblem::ScalingFormat::Vector)
                    {
                        mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
                    }
                    else
                    {
                        log_error(__func__, "invalid B scale mode", mode);
                        return rocblaslt_status_invalid_value;
                    }
                    memcpy(buf, &mode, sizeof(uint32_t));
                }
                break;
            case ROCBLASLT_MATMUL_DESC_POINTER_MODE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->pointermode, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->bias_type, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->amaxD, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->aux_type, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->compute_input_typeA, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->compute_input_typeB, sizeof(int32_t));
                break;
            default:
                log_error(__func__, "invalid attribute", matmulAttr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matmulDesc",
                    matmulDesc,
                    "attr",
                    rocblaslt_matmul_desc_attributes_to_string(matmulAttr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData[out]",
                    (void*)(uintptr_t)(*(uint32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_preference_create(rocblaslt_matmul_preference* pref)
{
    // Check if pref is valid
    if(pref == nullptr)
    {
        log_error(__func__, "invalid pointer", pref);
        return rocblaslt_status_invalid_handle;
    }
    *pref = nullptr;
    // Allocate
    try
    {
        *pref = new _rocblaslt_matmul_preference();
        log_api(__func__, "matmulPref[out]", *pref);
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_preference_destroy(const rocblaslt_matmul_preference pref)
{
    if(pref == nullptr)
    {
        log_error(__func__, "invalid pointer", pref);
        return rocblaslt_status_invalid_pointer;
    }

    log_api(__func__, "matmulPref", pref);
    // Destruct
    try
    {
        delete pref;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_preference_set_attribute(rocblaslt_matmul_preference            pref,
                                              rocblaslt_matmul_preference_attributes attribute,
                                              const void*                            data,
                                              size_t                                 dataSize)
{
    // Check if pref is valid
    if(data == nullptr || pref == nullptr)
    {
        log_error(__func__, "invalid pointer: data", data, "pref", pref);
        return rocblaslt_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        log_error(__func__, "invalid data size", dataSize);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        switch(attribute)
        {
        case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
            pref->search_mode = *(uint32_t*)data;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    dataSize,
                    "data",
                    pref->search_mode);
            break;
        case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            pref->max_workspace_bytes = *(uint64_t*)data;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    dataSize,
                    "data",
                    pref->max_workspace_bytes);
            break;
        case ROCBLASLT_MATMUL_PREF_OVERLAP_MODE_EXT:
            pref->overlap_mode = *(uint32_t*)data;
            break;
        case ROCBLASLT_MATMUL_PREF_EFFECTIVE_CU_COUNT_EXT:
            pref->effective_cu_count = *(uint32_t*)data;
            break;
        case ROCBLASLT_MATMUL_PREF_EFFECTIVE_CU_BUCKET_EXT:
            pref->effective_cu_bucket = *(uint32_t*)data;
            break;
        case ROCBLASLT_MATMUL_PREF_COMM_PRESSURE_LEVEL_EXT:
            pref->comm_pressure_level = *(uint32_t*)data;
            break;
        case ROCBLASLT_MATMUL_PREF_TOPOLOGY_SCOPE_EXT:
            pref->topology_scope = *(uint32_t*)data;
            break;
        case ROCBLASLT_MATMUL_PREF_POLICY_HINT_VERSION_EXT:
            pref->policy_hint_version = *(uint32_t*)data;
            break;
        default:
            log_error(__func__, "invalid attribute", attribute);
            return rocblaslt_status_invalid_value;
            break;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_preference_get_attribute(rocblaslt_matmul_preference            pref,
                                              rocblaslt_matmul_preference_attributes attribute,
                                              void*                                  data,
                                              size_t                                 sizeInBytes,
                                              size_t*                                sizeWritten)
{
    // Check if matmulDesc is valid
    if(data == nullptr || pref == nullptr)
    {
        log_error(__func__, "invalid pointer: data", data, "pref", pref);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid data size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        switch(attribute)
        {
        case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->search_mode;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    sizeInBytes,
                    "data[out]",
                    pref->search_mode);
            break;
        case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            *sizeWritten     = sizeof(uint64_t);
            *(uint64_t*)data = pref->max_workspace_bytes;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    sizeInBytes,
                    "data[out]",
                    pref->max_workspace_bytes);
            break;
        case ROCBLASLT_MATMUL_PREF_OVERLAP_MODE_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->overlap_mode;
            break;
        case ROCBLASLT_MATMUL_PREF_EFFECTIVE_CU_COUNT_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->effective_cu_count;
            break;
        case ROCBLASLT_MATMUL_PREF_EFFECTIVE_CU_BUCKET_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->effective_cu_bucket;
            break;
        case ROCBLASLT_MATMUL_PREF_COMM_PRESSURE_LEVEL_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->comm_pressure_level;
            break;
        case ROCBLASLT_MATMUL_PREF_TOPOLOGY_SCOPE_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->topology_scope;
            break;
        case ROCBLASLT_MATMUL_PREF_POLICY_HINT_VERSION_EXT:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->policy_hint_version;
            break;
        default:
            return rocblaslt_status_invalid_value;
            break;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/

rocblaslt_status rocblaslt_matmul_is_algo_supported(rocblaslt_handle        handle,
                                                    rocblaslt_matmul_desc   matmul_descr,
                                                    const void*             alpha,
                                                    rocblaslt_matrix_layout matA,
                                                    rocblaslt_matrix_layout matB,
                                                    const void*             beta,
                                                    rocblaslt_matrix_layout matC,
                                                    rocblaslt_matrix_layout matD,
                                                    rocblaslt_matmul_algo*  algo,
                                                    size_t*                 workspaceSizeInBytes)
{
    // Check if handle is valid
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }

    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        hipDataType            a_type       = matA->type;
        hipDataType            b_type       = matB->type;
        hipDataType            c_type       = matC->type;
        hipDataType            d_type       = matD->type;
        rocblaslt_compute_type compute_type = matmul_descr->compute_type;
        auto&                  gemmData     = matmul_descr->m_data;

        void* alphaf = (void*)alpha;
        void* betaf  = (void*)beta;
        auto  prob   = construct_rocblaslt_problem(
            handle, matmul_descr, matA, matB, matC, matD, alphaf, betaf, algo->max_workspace_bytes);
        status = isSolutionSupported(handle, prob, gemmData, algo, workspaceSizeInBytes);

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_algo_get_heuristic(rocblaslt_handle                  handle,
                                        rocblaslt_matmul_desc             matmul_desc,
                                        rocblaslt_matrix_layout           matA,
                                        rocblaslt_matrix_layout           matB,
                                        rocblaslt_matrix_layout           matC,
                                        rocblaslt_matrix_layout           matD,
                                        rocblaslt_matmul_preference       pref,
                                        int                               requestedAlgoCount,
                                        rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                        int*                              returnAlgoCount)
{
    // Check if handle is valid
    if(handle == nullptr || matmul_desc == nullptr || pref == nullptr || matA == nullptr
       || matB == nullptr || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid pointer");
        return rocblaslt_status_invalid_handle;
    }

    if(requestedAlgoCount < 1)
    {
        log_error(__func__, "invalid requested count", requestedAlgoCount);
        return rocblaslt_status_invalid_value;
    }
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        hipDataType            a_type       = matA->type;
        hipDataType            b_type       = matB->type;
        hipDataType            c_type       = matC->type;
        hipDataType            d_type       = matD->type;
        rocblaslt_compute_type compute_type = matmul_desc->compute_type;
        auto&                  tensile_data = matmul_desc->m_data;
        int8_t                 alpha[16]    = {0};
        int8_t                 beta[16]     = {0};
        assignAlphaBeta1(compute_type, (void*)alpha, (void*)beta);
        //bias ptr can be set later after getting solution.
        bool dummy_bias_address = false;
        if(matmul_desc->bias == nullptr && is_bias_enabled(matmul_desc->epilogue))
        {
            dummy_bias_address = true;
            matmul_desc->bias  = &dummy_bias_address;
        }
        auto prob = construct_rocblaslt_problem(
            handle, matmul_desc, matA, matB, matC, matD, &alpha, &beta, pref->max_workspace_bytes);
        applyUopcPreferenceToProblem(pref, prob);
        applyRuntimeUopcHint(handle, prob);

        ScopedOverrideSingleton& scopedExact      = ScopedOverrideSingleton::getInstance();
        OverrideSingleton&       override         = OverrideSingleton::getInstance();
        bool                     scoped_success   = false;
        bool                     override_success = false;
        if(scopedExact.env_mode && shouldUseScopedUopcExact(prob))
        {
            scoped_success = problem_scoped_override_from_file(handle,
                                                               prob,
                                                               matmul_desc,
                                                               heuristicResultsArray,
                                                               scopedExact.file_path,
                                                               pref->max_workspace_bytes);
            if(scoped_success)
                requestedAlgoCount--;

            log_api(__func__, "ScopedOverrideAlgoCount", scoped_success ? 1 : 0);
        }
        else if(override.env_mode)
        {
            override_success = problem_override_from_file(handle,
                                                          prob,
                                                          matmul_desc,
                                                          heuristicResultsArray,
                                                          override.file_path,
                                                          pref->max_workspace_bytes);
            if(override_success)
                requestedAlgoCount--;

            log_api(__func__, "OverrideAlgoCount", override_success ? 1 : 0);
        }

        if(requestedAlgoCount > 0)
        {
            status = getBestSolutions(prob,
                                      handle,
                                      tensile_data,
                                      requestedAlgoCount,
                                      (scoped_success || override_success)
                                          ? &heuristicResultsArray[1]
                                          : heuristicResultsArray,
                                      returnAlgoCount,
                                      pref->max_workspace_bytes);
        }

        if(scoped_success || override_success)
        {

            int oriReturnAlgoCount = *returnAlgoCount;
            if(!heuristicResult_check_duplicated(
                   &heuristicResultsArray[1], &heuristicResultsArray[0], oriReturnAlgoCount, true))
            {
                (*returnAlgoCount)++;
            }

            requestedAlgoCount++;
        }

        if(dummy_bias_address)
            matmul_desc->bias = nullptr;
        log_api(__func__, "returnAlgoCount", *returnAlgoCount);

        //Try to get size independent solutions from getAllSolutions()
        if(requestedAlgoCount > *returnAlgoCount)
        {
            std::vector<rocblaslt_matmul_heuristic_result> allSolutionsResults;
            if(rocblaslt_status_success
               == getAllSolutions(prob, handle, allSolutionsResults, pref->max_workspace_bytes))
            {
                int oriReturnAlgoCount = *returnAlgoCount;
                for(int i = 0;
                    *returnAlgoCount < requestedAlgoCount && i < allSolutionsResults.size();
                    i++)
                {
                    size_t required_workspace_size = 0;
                    if(heuristicResult_check_duplicated(heuristicResultsArray,
                                                        &allSolutionsResults[i],
                                                        oriReturnAlgoCount,
                                                        false)
                       || rocblaslt_status_success
                              != isSolutionSupported(handle,
                                                     prob,
                                                     tensile_data,
                                                     &allSolutionsResults[i].algo,
                                                     &required_workspace_size))
                        continue;

                    //append sol to heuristpicResultsArray
                    heuristicResult_copy(&heuristicResultsArray[*returnAlgoCount],
                                         &allSolutionsResults[i],
                                         pref->max_workspace_bytes,
                                         required_workspace_size);
                    (*returnAlgoCount)++;
                }

                log_api(__func__, "final returnAlgoCount", *returnAlgoCount);
            }
        }

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

#ifdef __cplusplus
}
#endif

void rocblaslt_init_gemmData(rocblaslt_handle       handle,
                             rocblaslt::RocGemmType gemmType,
                             hipblasOperation_t     opA,
                             hipblasOperation_t     opB,
                             hipDataType            typeA,
                             hipDataType            typeB,
                             hipDataType            typeC,
                             hipDataType            typeD,
                             rocblaslt_compute_type typeCompute,
                             size_t                 maxWorkspaceBytes,
                             std::shared_ptr<void>& gemmData)
{
    initTensileGemmData(handle,
                        gemmType,
                        opA,
                        opB,
                        typeA,
                        typeB,
                        typeC,
                        typeD,
                        typeCompute,
                        maxWorkspaceBytes,
                        gemmData);
}

rocblaslt_status rocblaslt_matmul_get_all_algos_cpp(
    rocblaslt_handle                                handle,
    rocblaslt::RocGemmType                          typeGemm,
    hipblasOperation_t                              opA,
    hipblasOperation_t                              opB,
    hipDataType                                     typeA,
    hipDataType                                     typeB,
    hipDataType                                     typeC,
    hipDataType                                     typeD,
    rocblaslt_compute_type                          typeCompute,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        log_error(__func__, "invalid pointer");
        return rocblaslt_status_invalid_handle;
    }
    // Create dummy
    auto initMat = [](_rocblaslt_matrix_layout& mat, hipDataType type) {
        mat.m    = 1;
        mat.n    = 1;
        mat.ld   = 1;
        mat.type = type;
    };
    _rocblaslt_matmul_desc   matmul_desc;
    _rocblaslt_matrix_layout matA;
    _rocblaslt_matrix_layout matB;
    _rocblaslt_matrix_layout matC;
    _rocblaslt_matrix_layout matD;
    initMat(matA, typeA);
    initMat(matB, typeB);
    initMat(matC, typeC);
    initMat(matD, typeD);
    matmul_desc.op_A                  = opA;
    matmul_desc.op_B                  = opB;
    matmul_desc.compute_type          = typeCompute;
    matmul_desc.scale_type            = typeD;
    rocblaslt_status status           = rocblaslt_status_success;
    size_t           maxWorkspaceSize = std::numeric_limits<size_t>::max();
    try
    {
        int8_t alpha[16] = {0};
        int8_t beta[16]  = {0};
        assignAlphaBeta1(matmul_desc.compute_type, (void*)alpha, (void*)beta);

        auto prob = construct_rocblaslt_problem(
            handle, &matmul_desc, &matA, &matB, &matC, &matD, &alpha, &beta, maxWorkspaceSize);
        if(typeGemm == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            status = getAllSolutions(prob, handle, heuristicResults, maxWorkspaceSize);
        }
        else if(typeGemm == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::vector<RocblasltContractionProblem> probs = {prob};
            status = getAllSolutions(probs, handle, heuristicResults, maxWorkspaceSize);
        }
        else
        {
            log_api(__func__, "Invalid gemm type", static_cast<int>(typeGemm));
            status = rocblaslt_status_not_implemented;
        }

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_matmul_get_algos_from_index_cpp(
    rocblaslt_handle                                handle,
    std::vector<int>&                               solutionIndex,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        size_t maxWorkspaceSize = std::numeric_limits<size_t>::max();
        status = getSolutionsFromIndex(handle, solutionIndex, heuristicResults, maxWorkspaceSize);

        log_api(__func__, "returnAlgoCount", heuristicResults.size());
        return status;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_is_algo_supported_cpp(rocblaslt_handle              handle,
                                                 rocblaslt::RocGemmType        gemmType,
                                                 std::shared_ptr<void>         gemmData,
                                                 rocblaslt_matmul_algo&        algo,
                                                 const rocblaslt::RocTuningV2* tuning,
                                                 size_t&                       workspaceSizeInBytes)
{
    return isSolutionSupported(handle, gemmType, gemmData, algo, tuning, workspaceSizeInBytes);
}

rocblaslt_status
    rocblaslt_algo_get_heuristic_cpp(rocblaslt_handle       handle,
                                     rocblaslt::RocGemmType gemmType,
                                     std::shared_ptr<void>  gemmData,
                                     const size_t           maxWorkspaceBytes,
                                     const int              requestedAlgoCount,
                                     std::vector<rocblaslt_matmul_heuristic_result>& results)
{
    if(requestedAlgoCount < 1)
    {
        log_error(__func__, "invalid requested count", requestedAlgoCount);
        return rocblaslt_status_invalid_value;
    }
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        log_api(
            __func__,
            "will be deprecated for groupedgemm in the future, please use get_all_algos instead");
    }
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        ScopedOverrideSingleton&                       scopedExact = ScopedOverrideSingleton::getInstance();
        OverrideSingleton&                             override = OverrideSingleton::getInstance();
        bool                                           scoped_success = false;
        bool                                           override_success = false;
        std::vector<rocblaslt_matmul_heuristic_result> override_result;

        if(scopedExact.env_mode && TensileDataGemmEffectiveCuCount(gemmData) > 0)
        {
            scoped_success = problem_scoped_override_from_file_cpp(
                handle, gemmType, gemmData, override_result, scopedExact.file_path, maxWorkspaceBytes);

            log_api(__func__, "ScopedOverrideAlgoCount", scoped_success ? 1 : 0);
        }
        else if(override.env_mode)
        {
            override_success = problem_override_from_file_cpp(
                handle, gemmType, gemmData, override_result, override.file_path, maxWorkspaceBytes);

            log_api(__func__, "OverrideAlgoCount", override_success ? 1 : 0);
        }

        if(requestedAlgoCount - override_result.size() > 0)
            status
                = getBestSolutions(handle,
                                   gemmType,
                                   gemmData,
                                   maxWorkspaceBytes,
                                   (scoped_success || override_success) ? requestedAlgoCount - 1
                                                                       : requestedAlgoCount,
                                   results);

        if(scoped_success || override_success)
        {

            results.insert(results.begin(), override_result[0]);
        }

        log_api(__func__, "returnAlgoCount", results.size());
        if(status != rocblaslt_status_success)
        {
            throw status;
        }
        //Try to get size independent solutions from getAllSolutions()
        if(requestedAlgoCount > results.size())
        {
            std::vector<rocblaslt_matmul_heuristic_result> allSolutionsResults;
            size_t                                         workspaceSizeInBytes = 0;
            if(rocblaslt_status_success
               == getAllSolutions(
                   gemmData, handle, gemmType, allSolutionsResults, maxWorkspaceBytes))
            {
                int oriReturnAlgoCount = results.size();
                for(int i = 0;
                    results.size() < requestedAlgoCount && i < allSolutionsResults.size();
                    i++)
                {
                    bool duplicated_sol = false;
                    for(int j = 0; j < oriReturnAlgoCount; j++)
                        if(*(int*)(results[j].algo.data)
                           == *(int*)(allSolutionsResults[i].algo.data)) //solution index
                            duplicated_sol = true;
                    rocblaslt::RocTuningV2* tuning = nullptr;
                    if(duplicated_sol == true
                       || rocblaslt_status_success
                              != isSolutionSupported(
                                  handle,
                                  static_cast<const rocblaslt::RocGemmType>(gemmType),
                                  gemmData,
                                  allSolutionsResults[i].algo,
                                  tuning,
                                  workspaceSizeInBytes))
                        continue;
                    allSolutionsResults[i].workspaceSize = workspaceSizeInBytes;
                    results.push_back(allSolutionsResults[i]);
                }

                log_api(__func__, "final returnAlgoCount", results.size());
            }
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_copy_matmul(rocblaslt_matmul_desc src, rocblaslt_matmul_desc dst)
{
    if(src == nullptr)
    {
        log_error(__func__, "invalid src matmulDescr pointer", src);
        return rocblaslt_status_invalid_pointer;
    }
    if(dst == nullptr)
    {
        log_error(__func__, "invalid dst matmulDescr pointer", dst);
        return rocblaslt_status_invalid_pointer;
    }
    dst->copy(*src);
    return rocblaslt_status_success;
}

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

struct ArchName
{
    std::string operator()(const hipDeviceProp_t& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

// exported. Get architecture name
std::string rocblaslt_internal_get_arch_name()
{
    int deviceId;
    static_cast<void>(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    return ArchName{}(deviceProperties);
}

bool rocblaslt_internal_test_path(const std::string& path)
{
#ifdef _WIN32
    return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

#ifdef _WIN32
std::string rocblaslt_internal_get_so_path()
{
    HMODULE hModule = NULL;
    if(!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                          // Should be the address of code in this library.
                          (LPCTSTR)rocblaslt_internal_get_so_path,
                          &hModule))
    {
        throw std::runtime_error("Cannot get module for function");
    }

    std::string path;
    path.resize(256);
    for(;;)
    {
        auto stored_size = GetModuleFileNameA(hModule, path.data(), path.size());
        if(stored_size < path.size())
        {
            // Success: size to what was stored (which does not include NUL).
            path.resize(stored_size);
            return path;
        }
        // Insufficient size.
        path.resize(path.size() * 2);
    }
}
#else
std::string rocblaslt_internal_get_so_path()
{
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(&rocblaslt_internal_get_so_path), &info) == 0)
    {
        throw std::runtime_error("Cannot get address of module function");
    }
    if(!info.dli_fname)
    {
        throw std::runtime_error("Containing binary does not have a file system path");
    }
    return std::string(info.dli_fname);
}
#endif

std::optional<std::filesystem::path> rocblaslt_find_library_relative_path(
    const std::optional<std::filesystem::path>& relpath,
    const std::optional<std::filesystem::path>& default_lib_dir)
{
    auto pathIfExists
        = [&](const std::filesystem::path& p) -> std::optional<std::filesystem::path> {
        if(relpath)
        {
            auto full_path = p / (*relpath);
            if(std::filesystem::exists(full_path))
                return full_path;
        }

        if(std::filesystem::exists(p))
            return p;
        return {};
    };

    auto probeLibDir
        = [&](const std::filesystem::path& lib_dir) -> std::optional<std::filesystem::path> {
        // There are a few fallback locations that have grown over time:
        //   {lib_dir}/hipblaslt/library
        // Legacy:
        //   {lib_dir}/../Tensile/library
        //   {lib_dir}/library
        if(auto p = pathIfExists(lib_dir / "hipblaslt" / "library"))
            return *p;
        if(auto p = pathIfExists(lib_dir.parent_path() / "Tensile" / "library"))
            return *p;
        if(auto p = pathIfExists(lib_dir / "library"))
            return *p;
        return std::nullopt;
    };

    if(default_lib_dir)
    {
        return probeLibDir(*default_lib_dir);
    }

    auto so_path       = std::filesystem::path(rocblaslt_internal_get_so_path()).parent_path();
    bool windows_style = false;
#ifdef _WIN32
    windows_style = true;
#endif

    // If on Windows, probe the sibling lib directory first, as that is non-deprecated.
    // Then fall back to the same-directory (bin) path.
    if(windows_style)
    {
        auto sibling = probeLibDir(so_path.parent_path() / "lib");
        if(sibling)
            return sibling;
    }

    return probeLibDir(so_path);
}

void rocblaslt_log_error(const char* func, const char* var, const char* msg)
{
    log_error(func, var, msg);
}

extern "C" int rocblaslt_matmul_is_tuned(rocblaslt_handle        handle,
                                         rocblaslt_matmul_desc   matmul_descr,
                                         rocblaslt_matrix_layout matA,
                                         rocblaslt_matrix_layout matB,
                                         rocblaslt_matrix_layout matC,
                                         rocblaslt_matrix_layout matD)
{
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return -1;
    }

    hipDataType            a_type              = matA->type;
    hipDataType            b_type              = matB->type;
    hipDataType            c_type              = matC->type;
    hipDataType            d_type              = matD->type;
    rocblaslt_compute_type compute_type        = matmul_descr->compute_type;
    auto&                  gemmData            = matmul_descr->m_data;
    float                  alpha               = 1.f;
    float                  beta                = 0.f;
    constexpr size_t       max_workspace_bytes = 32 * 1024 * 1024;
    void*                  alphaf              = &alpha;
    void*                  betaf               = &beta;
    auto                   prob                = construct_rocblaslt_problem(
        handle, matmul_descr, matA, matB, matC, matD, alphaf, betaf, max_workspace_bytes);
    auto& tensile_data = matmul_descr->m_data;
    auto  sols         = getBestRawSolutions(prob, handle, tensile_data, 1, max_workspace_bytes);

    if(sols.size() && sols.front()->tag == TensileLite::ContractionSolution::MatchingTag::Equal)
    {
        return 1;
    }

    return 0;
}
