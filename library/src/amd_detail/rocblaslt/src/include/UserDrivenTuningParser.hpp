/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#include "auxiliary.hpp"
#include "tensile_host.hpp"
#include <Tensile/DataTypes.hpp>
#include <shared_mutex>

#include <map>
#include <string>
#include <vector>

class OverrideSingleton
{
public:
    std::string file_path;
    bool        env_mode = false;

    static OverrideSingleton& getInstance()
    {
        static OverrideSingleton gInstance;
        return gInstance;
    }

    // copy contructor
    OverrideSingleton(const OverrideSingleton&) = delete;
    // assignment operator
    OverrideSingleton& operator=(const OverrideSingleton&) = delete;

private:
    OverrideSingleton()
    {
        char* Env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}
};

class ScopedOverrideSingleton
{
public:
    std::string file_path;
    bool        env_mode = false;

    static ScopedOverrideSingleton& getInstance()
    {
        static ScopedOverrideSingleton gInstance;
        return gInstance;
    }

    ScopedOverrideSingleton(const ScopedOverrideSingleton&) = delete;
    ScopedOverrideSingleton& operator=(const ScopedOverrideSingleton&) = delete;

private:
    ScopedOverrideSingleton()
    {
        char* Env = getenv("HIPBLASLT_UOPC_SCOPED_TUNING_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~ScopedOverrideSingleton() {}
};

namespace TensileLite
{

    enum class HeaderFields
    {
        transA = 0,
        transB,
        batch_count,
        m,
        n,
        k,
        a_type,
        b_type,
        c_type,
        compute_type,
        activation_type,
        bias_vector,
        bias_type,
        aux_type,
        solution_index,
        gcn_arch_name,
        cu_count,
        count
    };

    class ProblemOverride
    {
    public:
        ProblemOverride();
        ProblemOverride(bool             transA,
                        bool             transB,
                        rocisa::DataType inputTypeA,
                        rocisa::DataType inputTypeB,
                        rocisa::DataType computeType,
                        rocisa::DataType outputType,
                        size_t           m,
                        size_t           n,
                        size_t           k,
                        size_t           batchSize,
                        std::string      arch    = "",
                        size_t           cuCount = 0,
                        int              biasVector = 0,
                        rocisa::DataType biasType   = rocisa::DataType::None,
                        rocisa::DataType auxType    = rocisa::DataType::None,
                        std::string      activationType = "none");
        ProblemOverride(const ProblemOverride& problem);

        inline bool transA() const
        {
            return m_transA;
        }
        inline bool transB() const
        {
            return m_transB;
        }
        inline rocisa::DataType inputTypeA() const
        {
            return m_inputTypeA;
        }
        inline rocisa::DataType inputTypeB() const
        {
            return m_inputTypeB;
        }
        inline rocisa::DataType computeType() const
        {
            return m_computeType;
        }
        inline rocisa::DataType outputType() const
        {
            return m_outputType;
        }
        inline size_t m() const
        {
            return m_m;
        }
        inline size_t n() const
        {
            return m_n;
        }
        inline size_t k() const
        {
            return m_k;
        }
        inline size_t batchSize() const
        {
            return m_batchSize;
        }
        inline const std::string& arch() const
        {
            return m_arch;
        }
        inline size_t cuCount() const
        {
            return m_cuCount;
        }
        inline int biasVector() const
        {
            return m_biasVector;
        }
        inline rocisa::DataType biasType() const
        {
            return m_biasType;
        }
        inline rocisa::DataType auxType() const
        {
            return m_auxType;
        }
        inline const std::string& activationType() const
        {
            return m_activationType;
        }

    private:
        bool             m_transA;
        bool             m_transB;
        rocisa::DataType m_inputTypeA;
        rocisa::DataType m_inputTypeB;
        rocisa::DataType m_computeType;
        rocisa::DataType m_outputType;
        size_t           m_m;
        size_t           m_n;
        size_t           m_k;
        size_t           m_batchSize;
        std::string      m_arch;
        size_t           m_cuCount;
        int              m_biasVector;
        rocisa::DataType m_biasType;
        rocisa::DataType m_auxType;
        std::string      m_activationType;
    };

    std::pair<ProblemOverride, int> problemFromEntries(const std::vector<std::string>& entries);

    void getContractionProblemsFromFile(const std::string& path);

    template <>
    struct Comparison<ProblemOverride>
    {
        enum
        {
            implemented = true
        };

        static int compare(ProblemOverride const& lhs, ProblemOverride const& rhs)
        {
            return LexicographicCompare(lhs.arch(),
                                        rhs.arch(),
                                        lhs.cuCount(),
                                        rhs.cuCount(),
                                        lhs.transA(),
                                        rhs.transA(),
                                        lhs.transB(),
                                        rhs.transB(),
                                        lhs.inputTypeA(),
                                        rhs.inputTypeA(),
                                        lhs.inputTypeB(),
                                        rhs.inputTypeB(),
                                        lhs.computeType(),
                                        rhs.computeType(),
                                        lhs.outputType(),
                                        rhs.outputType(),
                                        lhs.m(),
                                        rhs.m(),
                                        lhs.n(),
                                        rhs.n(),
                                        lhs.k(),
                                        rhs.k(),
                                        lhs.batchSize(),
                                        rhs.batchSize(),
                                        lhs.activationType(),
                                        rhs.activationType(),
                                        lhs.biasVector(),
                                        rhs.biasVector(),
                                        lhs.biasType(),
                                        rhs.biasType(),
                                        lhs.auxType(),
                                        rhs.auxType());
        }
    };

    class OverrideMap
    {
    public:
        static OverrideMap& getMap()
        {
            static OverrideMap gInstance;
            return gInstance;
        }

        OverrideMap() {}
        ~OverrideMap() {}
        // copy contructor
        OverrideMap(const OverrideMap&) = delete;
        // assignment operator
        OverrideMap& operator=(const OverrideMap&) = delete;

        int size()
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      size = m_override.size();
            return size;
        }

        auto find(const ProblemOverride& prob_key)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      iter = m_override.equal_range(prob_key);
            return iter;
        }

        std::vector<int> lookup(const ProblemOverride& prob_key)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      range = m_override.equal_range(prob_key);
            std::vector<int>                          solutions;
            for(auto it = range.first; it != range.second; ++it)
            {
                solutions.push_back(it->second);
            }
            return solutions;
        }

        void add(const std::pair<ProblemOverride, int>& problemSolution)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.insert(problemSolution);
        }

        void erase(std::multimap<ProblemOverride, int>::iterator& sol_idx)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.erase(sol_idx);
        }

        std::mutex& getLock()
        {
            return m_guard;
        }

    private:
        std::multimap<ProblemOverride, int> m_override;
        std::mutex                          m_guard;
        std::shared_timed_mutex             m_mutex;
    };
} // namespace Tensile

namespace std
{
    template <>
    struct hash<TensileLite::ProblemOverride>
    {
        inline size_t operator()(TensileLite::ProblemOverride const& po) const
        {
            return TensileLite::hash_combine(po.arch(),
                                             po.cuCount(),
                                             po.transA(),
                                             po.transB(),
                                             po.inputTypeA(),
                                             po.inputTypeB(),
                                             po.computeType(),
                                             po.outputType(),
                                             po.m(),
                                             po.n(),
                                             po.k(),
                                             po.batchSize());
        }
    };
} // namespace std
