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

#include "UserDrivenTuningParser.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <shared_mutex>
#include <sstream>
#include <utility>

namespace TensileLite
{

    inline const char* HeaderFieldToString(HeaderFields field)
    {
        switch(field)
        {
        case HeaderFields::transA:
            return "transA";
        case HeaderFields::transB:
            return "transB";
        case HeaderFields::batch_count:
            return "batch_count";
        case HeaderFields::m:
            return "m";
        case HeaderFields::n:
            return "n";
        case HeaderFields::k:
            return "k";
        case HeaderFields::a_type:
            return "a_type";
        case HeaderFields::b_type:
            return "b_type";
        case HeaderFields::c_type:
            return "c_type";
        case HeaderFields::compute_type:
            return "compute_type";
        case HeaderFields::activation_type:
            return "activation_type";
        case HeaderFields::bias_vector:
            return "bias_vector";
        case HeaderFields::bias_type:
            return "bias_type";
        case HeaderFields::aux_type:
            return "aux_type";
        case HeaderFields::solution_index:
            return "solution_index";
        case HeaderFields::gcn_arch_name:
            return "gcnArchName";
        case HeaderFields::cu_count:
            return "CUs";
        default:
            return "";
        }
    }

    inline std::string normalizeArchName(const std::string& archName)
    {
        const auto first = archName.find_first_not_of(" \t\n\r\f\v");
        if(first == std::string::npos)
            return "";

        const auto last = archName.find_last_not_of(" \t\n\r\f\v");
        auto       arch = archName.substr(first, last - first + 1);
        const auto pos  = arch.find(':');
        if(pos != std::string::npos)
            arch.erase(pos);

        return arch;
    }

    inline std::string trimToken(const std::string& value)
    {
        const auto first = value.find_first_not_of(" \t\n\r\f\v");
        if(first == std::string::npos)
            return "";

        const auto last = value.find_last_not_of(" \t\n\r\f\v");
        return value.substr(first, last - first + 1);
    }

    inline std::string normalizeToken(const std::string& value)
    {
        auto normalized = trimToken(value);
        std::transform(normalized.begin(),
                       normalized.end(),
                       normalized.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return normalized;
    }

    inline bool parseHeaderField(const std::string& header, HeaderFields& outField)
    {
        const auto normalizedHeader = trimToken(header);
        for(size_t idx = 0; idx < static_cast<size_t>(HeaderFields::count); ++idx)
        {
            const auto field = static_cast<HeaderFields>(idx);
            if(normalizedHeader == HeaderFieldToString(field))
            {
                outField = field;
                return true;
            }
        }
        return false;
    }

    void getContractionProblemsFromFile(const std::string& path)
    {
        OverrideMap&                m_override = OverrideMap::getMap();
        std::mutex&                 map_guard  = m_override.getLock();
        std::lock_guard<std::mutex> lock(map_guard);

        if(m_override.size() == 0)
        {

            std::ifstream file_read(path);
            std::string   header_line, header;
            std::string   value_line, value;
            const auto    delim = ',';

            while(std::getline(file_read, header_line))
            {
                // Ignore lines without delimiter
                header_line.erase(0, header_line.find_first_not_of(" \t\n\r\f\v"));
                HeaderFields current_field = HeaderFields::transA;

                if(header_line.find(HeaderFieldToString(current_field)) != std::string::npos)
                {

                    if(std::getline(file_read, value_line))
                    {
                        value_line.erase(0, value_line.find_first_not_of(" \t\n\r\f\v"));
                        std::vector<std::string> entries(
                            static_cast<size_t>(HeaderFields::count));
                        std::stringstream header_split(header_line);
                        std::stringstream value_split(value_line);

                        while(std::getline(header_split, header, delim)
                              && std::getline(value_split, value, delim))
                        {
                            HeaderFields parsedField;
                            if(parseHeaderField(header, parsedField))
                            {
                                entries[static_cast<size_t>(parsedField)] = trimToken(value);
                            }
                        }

                        auto problemSolution = problemFromEntries(entries);

                        if(problemSolution.second > 0)
                        {
                            auto sol_iter       = m_override.find(problemSolution.first);
                            bool duplicate_find = false;

                            for(auto sol_idx = sol_iter.first; sol_idx != sol_iter.second;
                                sol_idx++)
                            {
                                if(sol_idx->second == problemSolution.second)
                                {
                                    duplicate_find = true;
                                    break;
                                }
                            }

                            if(!duplicate_find)
                            {
                                m_override.add(problemSolution);
                            }
                        }
                    }
                }
            }
        }
    }

    std::pair<ProblemOverride, int> problemFromEntries(const std::vector<std::string>& entries)
    {

        const size_t entries_n = entries.size();
        if(entries_n != static_cast<size_t>(HeaderFields::count))
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        bool transA = (normalizeToken(entries[static_cast<size_t>(HeaderFields::transA)]) != "n");
        bool transB = (normalizeToken(entries[static_cast<size_t>(HeaderFields::transB)]) != "n");

        size_t           m, n, b, k;
        rocisa::DataType inputTypeA  = rocisa::DataType::None;
        rocisa::DataType inputTypeB  = rocisa::DataType::None;
        rocisa::DataType outputType  = rocisa::DataType::None;
        rocisa::DataType computeType = rocisa::DataType::None;
        rocisa::DataType biasType    = rocisa::DataType::None;
        rocisa::DataType auxType     = rocisa::DataType::None;
        std::string      archName;
        std::string      activationType = "none";
        size_t           cuCount = 0;
        int              biasVector = 0;

        int solution_idx = -1;

        try
        {

            // TODO: are any additional mapping parameters needed?

            b = std::stol(trimToken(entries[static_cast<size_t>(HeaderFields::batch_count)]));
            m = std::stol(trimToken(entries[static_cast<size_t>(HeaderFields::m)]));
            n = std::stol(trimToken(entries[static_cast<size_t>(HeaderFields::n)]));
            k = std::stol(trimToken(entries[static_cast<size_t>(HeaderFields::k)]));
            inputTypeA = hipDataType_to_tensile_type(
                string_to_hip_datatype(trimToken(entries[static_cast<size_t>(HeaderFields::a_type)])));
            inputTypeB = hipDataType_to_tensile_type(
                string_to_hip_datatype(trimToken(entries[static_cast<size_t>(HeaderFields::b_type)])));
            outputType = hipDataType_to_tensile_type(
                string_to_hip_datatype(trimToken(entries[static_cast<size_t>(HeaderFields::c_type)])));
            computeType = rocComputeType_to_tensile_type(
                (rocblaslt_compute_type)string_to_hipblas_computetype(
                    trimToken(entries[static_cast<size_t>(HeaderFields::compute_type)])));

            const auto activationEntry
                = normalizeToken(entries[static_cast<size_t>(HeaderFields::activation_type)]);
            if(!activationEntry.empty())
                activationType = activationEntry;

            const auto biasVectorEntry
                = trimToken(entries[static_cast<size_t>(HeaderFields::bias_vector)]);
            if(!biasVectorEntry.empty())
                biasVector = std::stoi(biasVectorEntry);

            if(biasVector > 0)
            {
                const auto biasTypeEntry
                    = trimToken(entries[static_cast<size_t>(HeaderFields::bias_type)]);
                if(!biasTypeEntry.empty())
                {
                    biasType = hipDataType_to_tensile_type(
                        string_to_hip_datatype(biasTypeEntry));
                }
            }

            if(activationType != "none")
            {
                const auto auxTypeEntry = trimToken(entries[static_cast<size_t>(HeaderFields::aux_type)]);
                if(!auxTypeEntry.empty())
                {
                    auxType = hipDataType_to_tensile_type(string_to_hip_datatype(auxTypeEntry));
                }
            }

            solution_idx = std::stoi(trimToken(entries[static_cast<size_t>(HeaderFields::solution_index)]));

            const auto archEntry = trimToken(entries[static_cast<size_t>(HeaderFields::gcn_arch_name)]);
            if(!archEntry.empty())
            {
                archName = normalizeArchName(archEntry);
                const auto& cuCountEntry = entries[static_cast<size_t>(HeaderFields::cu_count)];
                if(!cuCountEntry.empty())
                    cuCount = std::stoul(trimToken(cuCountEntry));
            }
        }
        catch(std::invalid_argument const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        if(inputTypeA == rocisa::DataType::None || inputTypeB == rocisa::DataType::None
           || outputType == rocisa::DataType::None || computeType == rocisa::DataType::None)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        ProblemOverride po(
            transA,
            transB,
            inputTypeA,
            inputTypeB,
            computeType,
            outputType,
            m,
            n,
            k,
            b,
            archName,
            cuCount,
            biasVector,
            biasType,
            auxType,
            activationType);

        return std::make_pair(po, solution_idx);
    }

    ProblemOverride::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputTypeA(rocisa::DataType::None)
        , m_inputTypeB(rocisa::DataType::None)
        , m_computeType(rocisa::DataType::None)
        , m_outputType(rocisa::DataType::None)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
        , m_arch("")
        , m_cuCount(0)
        , m_biasVector(0)
        , m_biasType(rocisa::DataType::None)
        , m_auxType(rocisa::DataType::None)
        , m_activationType("none")
    {
    }

    ProblemOverride::ProblemOverride(bool             transA,
                                     bool             transB,
                                     rocisa::DataType inputTypeA,
                                     rocisa::DataType inputTypeB,
                                     rocisa::DataType computeType,
                                     rocisa::DataType outputType,
                                     size_t           m,
                                     size_t           n,
                                     size_t           k,
                                     size_t           batchSize,
                                     std::string      arch,
                                     size_t           cuCount,
                                     int              biasVector,
                                     rocisa::DataType biasType,
                                     rocisa::DataType auxType,
                                     std::string      activationType)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputTypeA(inputTypeA)
        , m_inputTypeB(inputTypeB)
        , m_computeType(computeType)
        , m_outputType(outputType)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)
        , m_arch(normalizeArchName(arch))
        , m_cuCount(cuCount)
        , m_biasVector(biasVector)
        , m_biasType(biasType)
        , m_auxType(auxType)
        , m_activationType(normalizeToken(activationType))
    {
    }

    ProblemOverride::ProblemOverride(const ProblemOverride& problem)
    {

        m_transA      = problem.transA();
        m_transB      = problem.transB();
        m_inputTypeA  = problem.inputTypeA();
        m_inputTypeB  = problem.inputTypeB();
        m_computeType = problem.computeType();
        m_outputType  = problem.outputType();
        m_m           = problem.m();
        m_n           = problem.n();
        m_k           = problem.k();
        m_batchSize   = problem.batchSize();
        m_arch        = problem.arch();
        m_cuCount     = problem.cuCount();
        m_biasVector  = problem.biasVector();
        m_biasType    = problem.biasType();
        m_auxType     = problem.auxType();
        m_activationType = problem.activationType();
    }

};
