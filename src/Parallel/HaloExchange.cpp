/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file HaloExchange.cpp
 * @brief Implementation of ambient parallel halo exchange using MPI
 *****************************************************************************/

// Standard library headers
#include <cstring>
#include <vector>

// External library headers
#include <mpi.h>

// Project headers
#include "HaloExchange.h"
#include "Comm.h"

// ***************************** Internal Storage *****************************

namespace
{

ProcessorPatchList g_processorPatches;
IndexList g_ghostGlobalIndices;

constexpr int haloTag = 50;

template<typename T>
void exchangeImpl(std::initializer_list<CellData<T>*> fields)
{
    if (g_processorPatches.empty() || !Comm::parallelRun())
    {
        return;
    }

    const Count numFields = fields.size();
    const Count numPatches = g_processorPatches.size();

    std::vector<std::vector<T>> sendBuffers(numPatches);
    std::vector<std::vector<T>> recvBuffers(numPatches);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * numPatches);

    // Receives first: layout is [field0 cells | field1 cells | ...]
    for (Index p = 0; p < numPatches; ++p)
    {
        recvBuffers[p].resize(numFields * g_processorPatches[p].ghostCellCount());

        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv
        (
            recvBuffers[p].data(),
            static_cast<int>(recvBuffers[p].size() * sizeof(T)),
            MPI_BYTE,
            static_cast<int>(g_processorPatches[p].neighborRank()),
            haloTag,
            MPI_COMM_WORLD,
            &request
        );
        requests.push_back(request);
    }

    // Sends
    for (Index p = 0; p < numPatches; ++p)
    {
        const IndexList& sendCells = g_processorPatches[p].sendCellIndices();
        sendBuffers[p].reserve(numFields * sendCells.size());

        for (const CellData<T>* field : fields)
        {
            for (const Index cellIdx : sendCells)
            {
                sendBuffers[p].push_back((*field)[cellIdx]);
            }
        }

        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Isend
        (
            sendBuffers[p].data(),
            static_cast<int>(sendBuffers[p].size() * sizeof(T)),
            MPI_BYTE,
            static_cast<int>(g_processorPatches[p].neighborRank()),
            haloTag,
            MPI_COMM_WORLD,
            &request
        );
        requests.push_back(request);
    }

    MPI_Waitall
    (
        static_cast<int>(requests.size()),
        requests.data(),
        MPI_STATUSES_IGNORE
    );

    // Unpack ghost cells
    for (Index p = 0; p < numPatches; ++p)
    {
        const Count ghostCount = g_processorPatches[p].ghostCellCount();
        const Index ghostFirst = g_processorPatches[p].ghostFirstCell();

        Index f = 0;
        for (CellData<T>* field : fields)
        {
            std::memcpy
            (
                field->data() + ghostFirst,
                recvBuffers[p].data() + f * ghostCount,
                ghostCount * sizeof(T)
            );
            ++f;
        }
    }
}

} // namespace (unnamed)

// ******************************* namespace Halo *****************************

namespace Halo
{

void init(ProcessorPatchList patches, IndexList ghostGlobalIndices)
{
    g_processorPatches = std::move(patches);
    g_ghostGlobalIndices = std::move(ghostGlobalIndices);
}

void reset() noexcept
{
    g_processorPatches.clear();
    g_ghostGlobalIndices.clear();
}

bool empty() noexcept
{
    return g_processorPatches.empty();
}

Count numPatches() noexcept
{
    return g_processorPatches.size();
}

const ProcessorPatchList& processorPatches() noexcept
{
    return g_processorPatches;
}

const IndexList& ghostGlobalIndices() noexcept
{
    return g_ghostGlobalIndices;
}

void exchange(std::initializer_list<ScalarField*> fields)
{
    exchangeImpl(fields);
}

void exchange(std::initializer_list<VectorField*> fields)
{
    exchangeImpl(fields);
}

void exchange(std::initializer_list<TensorField*> fields)
{
    exchangeImpl(fields);
}

} // namespace Halo
