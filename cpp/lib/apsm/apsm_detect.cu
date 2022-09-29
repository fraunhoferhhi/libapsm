/***
 *     ____  ____   __   _  _  __ _  _  _   __  ____  ____  ____    _  _  _  _  __
 *    (  __)(  _ \ / _\ / )( \(  ( \/ )( \ /  \(  __)(  __)(  _ \  / )( \/ )( \(  )
 *     ) _)  )   //    \) \/ (/    /) __ ((  O )) _)  ) _)  )   /  ) __ () __ ( )(
 *    (__)  (__\_)\_/\_/\____/\_)__)\_)(_/ \__/(__)  (____)(__\_)  \_)(_/\_)(_/(__)
 *     __ _  _  _  __  ____  __   __
 *    (  ( \/ )( \(  )(    \(  ) / _\
 *    /    /\ \/ / )(  ) D ( )( /    \
 *    \_)__) \__/ (__)(____/(__)\_/\_/
 *     ____   __    __  ___       ____   __  ____  ____
 *    (___ \ /  \  /  \/ _ \  ___(___ \ /  \(___ \(___ \
 *     / __/(  0 )(_/ /\__  )(___)/ __/(  0 )/ __/ / __/
 *    (____) \__/  (__)(___/     (____) \__/(____)(____)
 *
 * Copyright (c) 2019-2022
 * Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, including without
 * limitation the patents of Fraunhofer, ARE GRANTED BY THIS SOFTWARE LICENSE.
 * Fraunhofer provides no warranty of patent non-infringement with respect to
 * this software.
 */

/**
 * @file apsm_detect.cu
 * @brief APSM detect kernel
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 * @author Lukas Buse,        HHI, lukas.buse@hhi.fraunhofer.de
 *
 * @date 2019.11.25   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// APSM
#include "apsm/apsm_detect.cuh"
#include "apsm/apsm_noma_detector.cuh"
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_eventtimer.cuh"
#include "cuda/cuda_indexing.cuh"

// set namespace for cooperative groups
namespace cg = cooperative_groups;

// the warp size is very rarely changed, and it complicates reading the code.
// so we fix it with a define instead of setting it as a tune parameter in TuneKernel.
#define WARP_SIZE 32

/**
 * @brief Group tune parameters per function version
 * @details Different implementations of APSM Detect will have different operating points.
            This struct abstracts the tuning parameters so each version can have its own
            parameter set.
 */

template <int32_t version_id>
struct TuneKernel
{
};

/**************************************************************************************************
 * APSM_DETECT_ORIGINAL - Original
 *************************************************************************************************/

template <>
struct TuneKernel<APSM_DETECT_ORIGINAL>
{
    static const uint32_t BLOCK_BATCH_SIZE = 32;
};

/**
 * @brief CUDA shared device function of an linear APSM RKHS kernel
 *
 * @param[in] length vector length
 * @param[in] basis basis vector
 * @param[in] data data vector
 *
 * @return linear_inner_product
 */
__device__ RealSample kernel_apsm_detection_linear_original( uint32_t length, const RealSample* basis, const RealSample* data )
{
    // calculate inner product
    RealSample inner_product = RealSample( 0.0 );

    // NOTE: (mm) This for loop can be calculated in parallel,
    //            but the addition needs attention
#pragma unroll
    for ( uint32_t dim = 0; dim < length; dim++ )
    {

        inner_product += basis[ dim ] * data[ dim ];
    }

    // return linear kernel value
    return inner_product;
}

/**
 * @brief CUDA shared device function of an gaussian APSM RKHS kernel
 *
 * @param[in] length vector length
 * @param[in] basis basis vector
 * @param[in] data data vector
 * @param[in] variance variance value
 *
 * @return gaussian_inner_product
 */
__device__ RealSample kernel_apsm_detection_gaussian_original( uint32_t length, const CudaDeviceDedupRingBuffer& basis, uint32_t basisIdx, const RealSample* data, RealSample variance )
{

    // calculate weight
    RealSample exp_weight = RealSample( -0.5 ) / variance;

    // calculate argument
    RealSample exp_argument = RealSample( 0.0 );

    // NOTE: (mm) This for loop can be calculated in parallel,
    //            but the addition needs attention
#pragma unroll
    for ( uint32_t dim = 0; dim < length; dim++ )
    {

        RealSample dist_element = basis( basisIdx, dim ) - data[ dim ];
        exp_argument += dist_element * dist_element; // alternative: pow( dist_element, RealSample( 2.0 ) );
    }

    // return gaussian kernel value
    return exp( exp_weight * exp_argument );
}

/**
 * @brief CUDA shared detection device function
 * @details This is the original version called during the training phase.
 *
 * @param detectedSymbol
 * @param linearLength
 * @param basisLength
 * @param[in] rxdata_input pointer to intermediate memory
 * @param threadIdOffset
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[in] par APSM parameter set
 *
 * @return void
 */
__device__ void kernel_apsm_detection_original( RealSample* detectedSymbol, uint32_t linearLength, uint32_t gaussianLength, RealSample* rxdata_input, uint32_t threadIdOffset, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{

    // register for unique indexing
    // const uint32_t threadId	    = getThreadIdx_1D_1D();
    const uint32_t blockId = getBlockIdx_1D(); // gives sample id
    const uint32_t blockThreadId = getBlockThreadIdx_1D(); // gives basis id

    const uint32_t batch_size = getBlockDim_1D();

    const uint32_t basisLength = max( linearLength, gaussianLength );

    // ---------------------------------------------------------------------------

    // set register to zero
    if ( blockThreadId < batch_size )
        detectedSymbol[ blockThreadId ] = 0;

    // Iterate through the rx data vector
    // this for loop can be done in parallel and is computed in batches
    for ( uint32_t batch_idx = blockThreadId; batch_idx < linearLength; batch_idx += batch_size )
    {
        rxdata_input[ batch_idx ] = rx_data( batch_idx, threadIdOffset );
    }

    // we have to be sure that all threads finished, before we can use rxdata_input
    __syncthreads();

    // ---------------------------------------------------------------------------

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    for ( uint32_t basis_idx = blockThreadId; basis_idx < basisLength; basis_idx += batch_size )
    {
        // linear kernel and linear weight
        if ( basis_idx < linearLength )
        {
            // linear basis is identity matrix so kernel call is not necessary
            // furthermore basis does not contain linear part any more
            RealSample kernel_eval = par.linearKernelWeight * rxdata_input[ basis_idx ];
            kernel_eval *= trainingState.linearCoeffs( basis_idx );

            // because different threads writing to the same memory address
            atomicAdd( &detectedSymbol[ blockThreadId % batch_size ], kernel_eval );
        }
        // Gaussian kernel and Gaussian weight
        if ( basis_idx < gaussianLength )
        {
            // assign basis vector
            RealSample kernel_eval = par.gaussianKernelWeight * kernel_apsm_detection_gaussian_original( linearLength, trainingState.basis, basis_idx, rxdata_input, par.gaussianKernelVariance );
            kernel_eval *= trainingState.gaussianCoeffs( basis_idx );

            // because different threads writing to the same memory address
            atomicAdd( &detectedSymbol[ blockThreadId % batch_size ], kernel_eval );
        }
    }
}

/**
 * @brief CUDA detect kernel (original)
 * @details
 *
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[out] det_out equalized or decoded constallation vector
 * @param[in] par APSM parameter set
 *
 * @param[in] d_rxdata_input intermediate memory
 *
 * @return void
 */
template <>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_ORIGINAL>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // register for unique indexing
    const uint32_t blockId = getBlockIdx_1D(); // gives sample id
    const uint32_t blockThreadId = getBlockThreadIdx_1D(); // gives basis id

    const uint32_t batch_size = getBlockDim_1D();

    const uint32_t linearLength = rx_data.getHeight();

    const uint32_t gaussianLength = trainingState.basis.getUsedHeight();

    extern __shared__ RealSample shared[];

    // ---------------------------------------------------------------------------

    // Check that blockid is not outside of the range
    if ( blockId >= rx_data.getWidth() ) // data length
        return;

    // ---------------------------------------------------------------------------

    // Shared memory for detected symbol vector
    RealSample* detectedSymbol = &shared[ 0 ]; // detectedSymbol is manually set at the beginning of shared mem
    RealSample* rxdata_input = &shared[ batch_size ];

    // call shared detection function
    kernel_apsm_detection_original( detectedSymbol, linearLength, gaussianLength, rxdata_input,
                                    blockId, trainingState, rx_data, par );

    // we have to be sure that all threads finished, and than write it back
    __syncthreads();

    // only one thread per block writes output to GPU memory
    if ( blockThreadId == 0 )
    {
        for ( uint32_t idx = 1; idx < batch_size; idx++ )
            detectedSymbol[ 0 ] += detectedSymbol[ idx ];

        det_out( 0, blockId ) = detectedSymbol[ 0 ];
    }
}

/**************************************************************************************************
 * APSM_DETECT_OLDFAST - Old Fast
 *************************************************************************************************/

template <>
struct TuneKernel<APSM_DETECT_OLDFAST>
{
    static const uint32_t SAMPLES_PER_BLOCK = 16;
    static const uint32_t BASIS_PER_BATCH = 32;
    static const uint32_t PADDING = 0;
    static const uint32_t SHMEM_DETECTEDSYMBOL_SIZE = SAMPLES_PER_BLOCK * WARP_SIZE;
    static const uint32_t SHMEM_EXPARGUMENT_SIZE = SAMPLES_PER_BLOCK * BASIS_PER_BATCH;
    static const uint32_t SHMEM_BASISINPUT_SIZE = BASIS_PER_BATCH * ( WARP_SIZE + PADDING );
};

__device__ void kernel_apsm_detection_linear_oldfast( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, RealSample* detectedSymbol, uint32_t linearLength, uint32_t sample_idx, const CudaDeviceMatrix& coeff, const CudaDeviceDedupMatrix& rx_data, const RealSample linearKernelWeight )
{
    for ( uint32_t idx = tg.thread_rank(); idx < linearLength; idx += tg.size() )
    {
        const uint32_t local_idx = tg.thread_rank();

        RealSample kernel_eval = linearKernelWeight * rx_data( idx, sample_idx );
        RealSample coeff_val = coeff( idx );

        // because different threads writing to the same memory address
        atomicAdd( &detectedSymbol[ local_idx ], coeff_val * kernel_eval );
    }
}

__device__ void kernel_apsm_detection_gaussian_oldfast( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, RealSample* detectedSymbol, uint32_t linearLength, uint32_t basisLength, uint32_t sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_OLDFAST> TKP;

    __shared__ RealSample shmem_exp_argument[ TKP::SHMEM_EXPARGUMENT_SIZE ];
    __shared__ RealSample shmem_basis_input[ TKP::SHMEM_BASISINPUT_SIZE ];

    // set segment in share memory to use per sample (aka tile)
    RealSample* exp_argument = &shmem_exp_argument[ tg.meta_group_rank() * TKP::BASIS_PER_BATCH ];

#pragma unroll
    for ( uint32_t basis_offset = 0; basis_offset < basisLength; basis_offset += TKP::BASIS_PER_BATCH )
    {
#pragma unroll
        // reset exp_argument accumulations
        for ( uint32_t idx = tg.thread_rank(); idx < TKP::BASIS_PER_BATCH; idx += tg.size() )
        {
            exp_argument[ idx ] = RealSample( 0.0 );
        }

#pragma unroll
        for ( uint32_t dim = 0; dim < linearLength; dim += tg.size() )
        {
            // synchronize across the whole block to ensure all basis segments in the tile are fully read.
            cg::sync( cg::this_thread_block() );

            const uint32_t local_idx = tg.thread_rank();
            const uint32_t global_idx = dim + tg.thread_rank();
            RealSample rxdata_input_val = 0.0;

            if ( global_idx < linearLength )
                rxdata_input_val = rx_data( global_idx, sample_idx );

#pragma unroll
            // read the basis vector segments
            for ( uint32_t idx = tg.meta_group_rank(); idx < TKP::BASIS_PER_BATCH; idx += tg.meta_group_size() )
            {
                if ( global_idx < linearLength && basis_offset + idx < basisLength )
                    shmem_basis_input[ idx * ( WARP_SIZE + TKP::PADDING ) + local_idx ] = basis( basis_offset + idx, global_idx );
                else
                    shmem_basis_input[ idx * ( WARP_SIZE + TKP::PADDING ) + local_idx ] = 0.0;
            }
            // synchronize across the whole block to ensure all basis segments in the tile are fully read.
            cg::sync( cg::this_thread_block() );

#pragma unroll
            for ( uint32_t idx = 0; idx < TKP::BASIS_PER_BATCH; idx++ )
            {
                const uint32_t basis_idx = basis_offset + idx;

                if ( basis_idx >= basisLength )
                    continue;

                // prefetch the basis input from main memory, we will reuse them across samples
                RealSample* basis_input_current = &shmem_basis_input[ idx * ( WARP_SIZE + TKP::PADDING ) ];

                RealSample dist_element = basis_input_current[ local_idx ] - rxdata_input_val;
                RealSample elem2 = dist_element * dist_element;

#pragma unroll
                for ( uint32_t offset = tg.size() / 2; offset > 0; offset /= 2 )
                {
                    elem2 += tg.shfl_down( elem2, offset );
                }

                if ( tg.thread_rank() == 0 )
                {
                    exp_argument[ idx ] += elem2;
                }
            }
        }
        tg.sync();

        const RealSample exp_weight = RealSample( -0.5 ) / par.gaussianKernelVariance;
        const RealSample gaussianKernelWeight = par.gaussianKernelWeight;

#pragma unroll
        // partial accumulations are done, now finish the kernel evaluation for the current basis batch.
        for ( uint32_t basis_idx = basis_offset + tg.thread_rank(); basis_idx < basis_offset + TKP::BASIS_PER_BATCH; basis_idx += tg.size() )
        {
            const uint32_t local_idx = tg.thread_rank();

            if ( basis_idx >= basisLength )
                continue;

            RealSample kernel_eval = gaussianKernelWeight * exp( exp_weight * exp_argument[ local_idx ] );
            RealSample coeff_val = coeff( basis_idx );

            // because different threads writing to the same memory address
            atomicAdd( &detectedSymbol[ local_idx ], coeff_val * kernel_eval );
        }
    }
}

/**
 * @brief CUDA shared detection device function
 * @details Fast version.
 *
 * @param detectedSymbol
 * @param linearLength
 * @param basisLength
 * @param[in] rxdata_input pointer to intermediate memory
 * @param threadIdOffset
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[in] par APSM parameter set
 *
 * @return void
 */
__device__ void kernel_apsm_detection_oldfast( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const uint32_t sample_idx, RealSample& detSymbol, const uint32_t linearLength, const uint32_t basisLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_OLDFAST> TKP;

    __shared__ RealSample shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    RealSample* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];

    detectedSymbol[ tg.thread_rank() ] = RealSample( 0.0 );

    kernel_apsm_detection_linear_oldfast( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_oldfast( bg, tg, detectedSymbol, linearLength, basisLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce the detected symbols to a single one.
    RealSample acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();

#pragma unroll
    for ( uint32_t idx = tg.size() / 2; idx > 0; idx /= 2 )
    {
        acc += tg.shfl_down( acc, idx );
    }
    if ( tg.thread_rank() == 0 )
    {
        detSymbol = acc;
    }
    return;
}

/**
 * @brief CUDA detect kernel (fast)
 * @details This is an improved version of the detection kernel
 *
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[out] det_out equalized or decoded constallation vector
 * @param[in] par APSM parameter set
 *
 * @param[in] d_rxdata_input intermediate memory
 *
 * @return void
 */
template <>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_OLDFAST>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_OLDFAST> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const uint32_t blockId = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const uint32_t sample_idx = TKP::SAMPLES_PER_BLOCK * blockId + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // TODO: can this happen? or is it an error we should check before launch?
    // Check that blockid is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const uint32_t linearLength = rx_data.getHeight();
    const uint32_t basisLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol
    RealSample detected_symbol = RealSample( 0.0 );

    kernel_apsm_detection_oldfast( block,
                                   sample_block,
                                   sample_idx,
                                   detected_symbol,
                                   linearLength,
                                   basisLength,
                                   trainingState,
                                   rx_data,
                                   par );

    // write result to global memory
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}

/**************************************************************************************************
 * APSM_DETECT_SHMEM - Store vectors in shared memory
 *************************************************************************************************/

template <>
struct TuneKernel<APSM_DETECT_SHMEM>
{
    static const uint32_t SAMPLES_PER_BLOCK = 4;
    static const uint32_t MAX_LINEAR_LENGTH = 64;
    static const uint32_t PADDING = 1;
    static const uint32_t PADDED_LINEAR_LENGTH = MAX_LINEAR_LENGTH + PADDING;
    static const uint32_t BASIS_PER_BATCH = WARP_SIZE;
    static const uint32_t SHMEM_DETECTEDSYMBOL_SIZE = SAMPLES_PER_BLOCK * WARP_SIZE;
    static const uint32_t SHMEM_RXDATA_SIZE = PADDED_LINEAR_LENGTH * SAMPLES_PER_BLOCK;
    static const uint32_t SHMEM_BASIS_SIZE = PADDED_LINEAR_LENGTH * BASIS_PER_BATCH;
};

__device__ void kernel_apsm_detection_linear_shmem( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, RealSample* detectedSymbol, uint32_t linearLength, uint32_t sample_idx, const CudaDeviceMatrix& coeff, const CudaDeviceDedupMatrix& rx_data, const RealSample linearKernelWeight )
{
    for ( uint32_t idx = tg.thread_rank(); idx < linearLength; idx += tg.size() )
    {
        const uint32_t local_idx = tg.thread_rank();

        RealSample kernel_eval = linearKernelWeight * rx_data( idx, sample_idx );
        RealSample coeff_val = coeff( idx );

        // because different threads writing to the same memory address
        atomicAdd( &detectedSymbol[ local_idx ], coeff_val * kernel_eval );
    }
}

__device__ void kernel_apsm_detection_gaussian_shmem( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, RealSample* detectedSymbol, uint32_t linearLength, uint32_t basisLength, uint32_t sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    const uint32_t tid = tg.thread_rank();
    const RealSample exp_weight = RealSample( -0.5 ) / par.gaussianKernelVariance;

    // Padding is used in the following shared memory arrays to avoid bank conflicts.

    // each tile can cache the input data in share memory
    __shared__ RealSample shmem_rxdata[ TKP::SHMEM_RXDATA_SIZE ];
    RealSample* data = &shmem_rxdata[ tg.meta_group_rank() * TKP::PADDED_LINEAR_LENGTH ];

    // each thread can cache a basis vector, and share it across other samples in the block
    __shared__ RealSample shmem_basis[ TKP::SHMEM_BASIS_SIZE ];
    RealSample* basis_sh = &shmem_basis[ tg.thread_rank() * TKP::PADDED_LINEAR_LENGTH ];

    // read rxdata into shared memory
    for ( uint32_t idx = tg.thread_rank(); idx < linearLength; idx += tg.size() )
        data[ idx ] = rx_data( idx, sample_idx );
    tg.sync();

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    // tg.size() and BASIS_PER_BATCH must match, otherwise not all basis are processed. Here both are WARP_SIZE.
    for ( uint32_t basis_offset = 0; basis_offset < basisLength; basis_offset += TKP::BASIS_PER_BATCH )
    {
        // wait before reading new data, avoid overwrite of data being processed by previous iteration.
        bg.sync();

        const uint32_t basis_idx = basis_offset + tg.thread_rank();

        // let the first tile (sample) in the block issue the data read operations.
        if ( ( tg.meta_group_rank() == 0 ) && ( basis_idx < basisLength ) )
        {
            for ( uint32_t idx = 0; idx < linearLength; idx++ )
                basis_sh[ idx ] = basis( basis_idx, idx );
        }
        // tell all tiles (samples) to wait for data to be available.
        bg.sync();

        // since we are advancing in batches, it is possible that a basis index
        // goes above the limit. In that case, skip the rest of the loop.
        // Note that we do this after the sync of the block.
        if ( basis_idx >= basisLength )
            continue;

        // Gaussian kernel and Gaussian weight
        // Embed gaussian kernel in loop, as we will now process the argument in batches
        RealSample exp_argument = RealSample( 0.0 );
        for ( uint32_t dim = 0; dim < linearLength; dim++ )
        {
            RealSample dist_element = basis_sh[ dim ] - data[ dim ];
            exp_argument += dist_element * dist_element;
        }
        RealSample kernel_eval = coeff( basis_idx ) * par.gaussianKernelWeight * exp( exp_weight * exp_argument );

        // because different threads writing to the same memory address
        atomicAdd( &detectedSymbol[ tid ], kernel_eval );
    }
}

__device__ void kernel_apsm_detection_shmem( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const uint32_t sample_idx, RealSample& detSymbol, const uint32_t linearLength, const uint32_t gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    // work over a tile, that does work for a single sample.
    __shared__ RealSample shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    RealSample* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];

    const uint32_t tid = tg.thread_rank();

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // compute linear and gaussian contributions
    kernel_apsm_detection_linear_shmem( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_shmem( bg, tg, detectedSymbol, linearLength, gaussianLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce symbols
    RealSample acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();
    for ( uint32_t idx = tg.size() / 2; idx > 0; idx /= 2 )
        acc += tg.shfl_down( acc, idx );

    if ( tg.thread_rank() == 0 )
        detSymbol = acc;
}

template <>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_SHMEM>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const uint32_t block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const uint32_t sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const uint32_t linearLength = rx_data.getHeight();
    const uint32_t gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    RealSample detected_symbol = RealSample( 0.0 );

    kernel_apsm_detection_shmem( block,
                                 sample_block,
                                 sample_idx,
                                 detected_symbol,
                                 linearLength,
                                 gaussianLength,
                                 trainingState,
                                 rx_data,
                                 par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}

/**************************************************************************************************
 * APSM_DETECT_BALANCED - Balance computation and memory accesses
 *************************************************************************************************/

template <>
struct TuneKernel<APSM_DETECT_BALANCED>
{
    static const uint32_t SAMPLES_PER_BLOCK = 32;
    static const uint32_t PADDING = 1;
    static const uint32_t BASIS_PER_BATCH = 4 * WARP_SIZE;
    static const uint32_t SHMEM_DETECTEDSYMBOL_SIZE = SAMPLES_PER_BLOCK * WARP_SIZE;
    static const uint32_t SHMEM_RXDATA_SIZE = ( WARP_SIZE + PADDING ) * SAMPLES_PER_BLOCK;
    static const uint32_t SHMEM_BASIS_SIZE = ( WARP_SIZE + PADDING ) * BASIS_PER_BATCH;
    static const uint32_t SHMEM_EXPARGUMENT_SIZE = SAMPLES_PER_BLOCK * ( BASIS_PER_BATCH + PADDING );
};

// function pointer to link linear balanced function to linear shmem function
typedef void ( *linearFunctionPointer_t )( cg::thread_block&, cg::thread_block_tile<WARP_SIZE>&, RealSample*, uint32_t, uint32_t, const CudaDeviceMatrix&, const CudaDeviceDedupMatrix&, const RealSample );
__device__ linearFunctionPointer_t kernel_apsm_detection_linear_balanced = kernel_apsm_detection_linear_shmem;

__device__ void kernel_apsm_detection_gaussian_balanced( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, RealSample* detectedSymbol, uint32_t linearLength, uint32_t gaussianLength, uint32_t sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    const uint32_t tid = tg.thread_rank();
    const RealSample exp_weight = RealSample( -0.5 ) / par.gaussianKernelVariance;

    // each tile can cache a segment (WARP_SIZE) of the input data in share memory
    __shared__ RealSample shmem_rxdata[ TKP::SHMEM_RXDATA_SIZE ];
    RealSample* data = &shmem_rxdata[ tg.meta_group_rank() * ( WARP_SIZE + TKP::PADDING ) ];

    // each block can cache a batch of segments (WARP_SIZE) of basis vectors, and share it across other samples in the block
    __shared__ RealSample shmem_basis[ TKP::SHMEM_BASIS_SIZE ];

    // each tile keeps intermediate values for the exp_arguments in shared memory (as many as in a BASIS_PER_BATCH)
    __shared__ RealSample shmem_exp_argument[ TKP::SHMEM_EXPARGUMENT_SIZE ];
    RealSample* exp_argument = &shmem_exp_argument[ tg.meta_group_rank() * ( TKP::BASIS_PER_BATCH + TKP::PADDING ) ];

    // Iterate through the dictionary, in batches
    for ( uint32_t basis_offset = 0; basis_offset < gaussianLength; basis_offset += TKP::BASIS_PER_BATCH )
    {
        tg.sync();
        // clear exp_argument for all elements in the batch
        for ( uint32_t idx = tid; idx < TKP::BASIS_PER_BATCH; idx += WARP_SIZE )
            exp_argument[ idx ] = RealSample( 0.0 );

        // traverse the vector length in segments of WARP_SIZE, makes the algorithm independent of linear length
        for ( uint32_t dim_offset = 0; dim_offset < max( linearLength, WARP_SIZE ); dim_offset += WARP_SIZE )
        {
            const RealSample dim_oob = ( dim_offset + tid < linearLength ) ? 1.0 : 0.0;

            // wait before reading new data, avoid overwrite of data being processed by previous iteration.
            bg.sync();

            // load the data input segment
            data[ tid ] = dim_oob * rx_data( dim_offset + tid, sample_idx );

            // load the basis vectors segments

// v1: fetch from single tile
//     use a single tile in the block, similar to SHMEM
#if 0
            if ( tg.meta_group_rank() == 0 )
            {
#pragma unroll
                for ( uint32_t bid = tid; bid < TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
                {
                    const uint32_t basis_idx = basis_offset + bid;
                    const RealSample basis_oob = ( basis_idx < gaussianLength ) ? 1.0 : 0.0;
                    RealSample* basis_sh = &shmem_basis[ bid * ( WARP_SIZE + TKP::PADDING ) ];

#pragma unroll
                    for ( uint32_t idx = 0; idx < WARP_SIZE; idx++ )
                    {
                        const RealSample dim_oob2 = ( dim_offset + idx < linearLength ) ? 1.0 : 0.0;
                        basis_sh[ idx ] = basis_oob * dim_oob2 * basis( basis_idx, dim_offset + idx );
                    }
                }
            }
#endif

// v2: fetch using all warps in the block
#if 1
            {
#pragma unroll
                for ( uint32_t bid = tid; bid < TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
                {
                    const uint32_t basis_idx = basis_offset + bid;
                    if ( basis_idx >= gaussianLength )
                        break;

                    RealSample* basis_sh = &shmem_basis[ bid * ( WARP_SIZE + TKP::PADDING ) ];

#pragma unroll
                    for ( uint32_t idx = tg.meta_group_rank(); idx < WARP_SIZE; idx += tg.meta_group_size() )
                    {
                        const RealSample dim_oob2 = ( dim_offset + idx < linearLength ) ? 1.0 : 0.0;
                        basis_sh[ idx ] = dim_oob2 * basis( basis_idx, dim_offset + idx );
                    }
                }
            }
#endif
            // tell all tiles (samples) to wait for data to be available.
            bg.sync();

// compute stage, part 1
// Process the batch for this segment and save the intermediate result in shared memory
#pragma unroll
            for ( uint32_t bid = tid; bid < TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
            {
                RealSample* basis_sh = &shmem_basis[ bid * ( WARP_SIZE + TKP::PADDING ) ];
#pragma unroll
                for ( uint32_t dim = 0; dim < min( WARP_SIZE, linearLength ); dim++ )
                {
                    RealSample dist_element = basis_sh[ dim ] - data[ dim ];
                    exp_argument[ bid ] += dist_element * dist_element;
                }
            }
        }
        tg.sync();

// compute stage, part 2: now that all contributions in exp_argument are complete,
// finalize the detected symbol computation, again going over the whole batch
#pragma unroll
        for ( uint32_t bid = tid; bid < TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
        {
            const uint32_t basis_idx = basis_offset + bid;

            if ( basis_idx >= gaussianLength )
                continue;

            RealSample kernel_eval = coeff( basis_idx ) * par.gaussianKernelWeight * exp( exp_weight * exp_argument[ bid ] );

            // because different threads writing to the same memory address
            atomicAdd( &detectedSymbol[ tid ], kernel_eval );
        }
    }
}

__device__ void kernel_apsm_detection_balanced( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const uint32_t sample_idx, RealSample& detSymbol, const uint32_t linearLength, const uint32_t gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    // work over a tile, that does work for a single sample.

    __shared__ RealSample shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    RealSample* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];

    const uint32_t tid = tg.thread_rank();

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // compute linear and gaussian contributions
    kernel_apsm_detection_linear_balanced( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_balanced( bg, tg, detectedSymbol, linearLength, gaussianLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce symbols
    RealSample acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();
    for ( uint32_t idx = tg.size() / 2; idx > 0; idx /= 2 )
        acc += tg.shfl_down( acc, idx );

    if ( tg.thread_rank() == 0 )
        detSymbol = acc;
}

template <>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_BALANCED>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const uint32_t block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const uint32_t sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const uint32_t linearLength = rx_data.getHeight();
    const uint32_t gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    RealSample detected_symbol = RealSample( 0.0 );

    kernel_apsm_detection_balanced( block,
                                    sample_block,
                                    sample_idx,
                                    detected_symbol,
                                    linearLength,
                                    gaussianLength,
                                    trainingState,
                                    rx_data,
                                    par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}

/**************************************************************************************************
 * APSM Wrapper Function - Entry point to the detect function from the wrapper code
 *************************************************************************************************/

/**
 * @brief C++ wrapper function for APSM detection part (wrapper specialization for APSM_DETECT_ORIGINAL version).
 * @details This function call the APSM CUDA detect kernel.
 *
 * @param[in] trainingState contains basis matrix (dictionary) as well as linear and gaussian coefficients learned in train
 * @param[in] d_apsm_rxd2r received rx constallation vector
 * @param[out] d_apsm_esd2r equalized or decoded constallation vector
 *
 * @return float measured kernel processing time
 */
template <>
float ApsmNomaDetector<apsm_versions::APSM_DETECT_ORIGINAL>::wrapperDetect( const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_ORIGINAL> TKP;

    // Initialise timer
    CUDA_EventTimer timer;

    // get vector sizes
    uint32_t vector_size = d_apsm_rxd2r.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate shared memory size parameter
    uint32_t sharedMemorySize = 0;

    // NOTE: for this version we use shared memory
    sharedMemorySize += TKP::BLOCK_BATCH_SIZE * sizeof( RealSample );
    sharedMemorySize += d_apsm_rxd2r.getHeight() * sizeof( RealSample );

    // run kernel and measure time
    timer.start( stream );
    kernel_apsm_detect<apsm_versions::APSM_DETECT_ORIGINAL><<<grid_dim, block_dim, sharedMemorySize, stream>>>( trainingState->toDevice(), d_apsm_rxd2r.toDevice(), d_apsm_esd2r.toDevice(), par );
    timer.stop( stream );

    // debug print
    // printf( "kernel name    : kernel_apsm_detect\n" );
    // printf( "kernel version : %s\n", apsm_get_version_string( version_id ).c_str() );
    // printf( "Grid size      : [ %i, %i, %i ]\n", grid_dim.x, grid_dim.y, grid_dim.z ) ;
    // printf( "Block size     : [ %i, %i, %i ]\n", block_dim.x, block_dim.y, block_dim.z ) ;
    // printf( "SHMEM size     : %i\n", sharedMemorySize );
    // std::cout << "APSM parameters : "
    //        << par << std::endl;

    // check for errors after cuda api calls
    CUDA_CHECK_ERROR(); // TODO: (mm) this shows "too many resources requested for launch error"
        //            in DEBUG mode and for 2 RX antennas
        //            with gaussian weight = 1 we see two times the same value in detSigData

    // give back the measured kernel processing time
    return timer.elapsed();
}

/**
 * @brief C++ wrapper function for APSM detection part.
 * @details This function call the APSM CUDA detect kernel.
 *
 * @param[in] trainingState contains basis matrix (dictionary) as well as linear and gaussian coefficients learned in train
 * @param[in] d_apsm_rxd2r received rx constallation vector
 * @param[out] d_apsm_esd2r equalized or decoded constallation vector
 *
 * @return float measured kernel processing time
 */
template <int32_t version_id>
float ApsmNomaDetector<version_id>::wrapperDetect( const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r )
{
    // rename type for easier reference
    typedef struct TuneKernel<version_id> TKP;

    // Initialise timer
    CUDA_EventTimer timer;

    // get vector sizes
    uint32_t vector_size = d_apsm_rxd2r.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate dynamic shared memory size parameter
    uint32_t sharedMemorySize = 0;

    // NOTE: for this version we fix launch dimensions
    grid_dim.x = ( vector_size + TKP::SAMPLES_PER_BLOCK - 1 ) / TKP::SAMPLES_PER_BLOCK;
    block_dim.x = TKP::SAMPLES_PER_BLOCK * WARP_SIZE;

    // run kernel and measure time
    timer.start( stream );
    kernel_apsm_detect<version_id><<<grid_dim, block_dim, sharedMemorySize, stream>>>( trainingState->toDevice(), d_apsm_rxd2r.toDevice(), d_apsm_esd2r.toDevice(), par );
    timer.stop( stream );

    // debug print
    // printf( "kernel name    : kernel_apsm_detect\n" );
    // printf( "kernel version : %s\n", apsm_get_version_string( version_id ).c_str() );
    // printf( "Grid size      : [ %i, %i, %i ]\n", grid_dim.x, grid_dim.y, grid_dim.z ) ;
    // printf( "Block size     : [ %i, %i, %i ]\n", block_dim.x, block_dim.y, block_dim.z ) ;
    // printf( "SHMEM size     : %i\n", sharedMemorySize );
    // std::cout << "APSM parameters : "
    //        << par << std::endl;

    // check for errors after cuda api calls
    CUDA_CHECK_ERROR(); // TODO: (mm) this shows "too many resources requested for launch error"
        //            in DEBUG mode and for 2 RX antennas
        //            with gaussian weight = 1 we see two times the same value in detSigData

    // give back the measured kernel processing time
    return timer.elapsed();
}

// explicit instantiation of all known versions of the wrapper, so they are present in the library
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_ORIGINAL>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_OLDFAST>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_SHMEM>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_BALANCED>;
