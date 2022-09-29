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

// STD C
#include <cassert>
#include <fstream>
#include <iostream>

// APSM
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_indexing.cuh"
#include "noma/noma_train_history.cuh"

NomaTrainHistory::NomaTrainHistory( uint32_t bitPerSymbol, float modulationScale )
    : bitPerSymbol( bitPerSymbol )
    , modulationScale( modulationScale )
{
    CUDA_CHECK( cudaStreamCreate( &stream ) );
}

NomaTrainHistory::~NomaTrainHistory()
{
    CUDA_CHECK( cudaStreamDestroy( stream ) );
}

void NomaTrainHistory::addTrainStep( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    evmTrain.push_back( computeEvm( y_true, y_hat, numElements ) );
    serTrain.push_back( computeSer( y_true, y_hat, numElements ) );
    berTrain.push_back( computeBer( y_true, y_hat, numElements ) );
}

void NomaTrainHistory::addValStep( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    evmVal.push_back( computeEvm( y_true, y_hat, numElements ) );
    serVal.push_back( computeSer( y_true, y_hat, numElements ) );
    berVal.push_back( computeBer( y_true, y_hat, numElements ) );
}

__device__ void sumResults( RealSample* shmem, RealSample& target, const uint32_t blockThreadId )
{
    __syncthreads();

    for ( uint32_t size = APSM_BLOCK_BATCH_SIZE / 2; size > 0; size /= 2 )
    {
        if ( blockThreadId < size )
            shmem[ blockThreadId ] += shmem[ blockThreadId + size ];
        __syncthreads();
    }

    if ( blockThreadId == 0 )
        target = shmem[ 0 ];
}

RealSample NomaTrainHistory::getResult( const void* resultSymbol )
{
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    RealSample result = 0;
    CUDA_CHECK( cudaMemcpyFromSymbol( &result, resultSymbol, sizeof( RealSample ) ) );
    return result;
}

__device__ RealSample evmReturn;

__global__ void kernelComputeEvm( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    __shared__ RealSample sumShmem[ APSM_BLOCK_BATCH_SIZE ];
    sumShmem[ blockThreadId ] = 0;

    for ( uint32_t idx = blockThreadId; idx < numElements / 2; idx += batchSize )
    {
        RealSample reDiff = y_true[ 2 * idx ] - y_hat[ 2 * idx ];
        RealSample imDiff = y_true[ 2 * idx + 1 ] - y_hat[ 2 * idx + 1 ];
        sumShmem[ blockThreadId ] += reDiff * reDiff + imDiff * imDiff;
    }

    sumResults( sumShmem, evmReturn, blockThreadId );
}

RealSample NomaTrainHistory::computeEvm( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    kernelComputeEvm<<<1, APSM_BLOCK_BATCH_SIZE, 0, stream>>>( y_true, y_hat, numElements );
    return sqrt( getResult( &evmReturn ) / ( numElements / 2 ) );
}

__device__ RealSample serReturn;

__device__ bool symbolError1D( RealSample y_true, RealSample y_hat, RealSample modScale, RealSample maxConstellationVal )
{
    return fabsf( y_true - y_hat ) > modScale
        && !( y_hat > +maxConstellationVal && y_true > +maxConstellationVal - modScale )
        && !( y_hat < -maxConstellationVal && y_true < -maxConstellationVal + modScale );
}

__global__ void kernelComputeSer( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements, RealSample modScale, RealSample maxConstellationVal )
{
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    __shared__ RealSample sumShmem[ APSM_BLOCK_BATCH_SIZE ];
    sumShmem[ blockThreadId ] = 0;

    for ( uint32_t idx = blockThreadId; idx < numElements / 2; idx += batchSize )
    {
        bool symbolError = symbolError1D( y_true[ 2 * idx ], y_hat[ 2 * idx ], modScale, maxConstellationVal )
            || symbolError1D( y_true[ 2 * idx + 1 ], y_hat[ 2 * idx + 1 ], modScale, maxConstellationVal );
        sumShmem[ blockThreadId ] += symbolError;
    }

    sumResults( sumShmem, serReturn, blockThreadId );
}

__global__ void kernelComputeSerBpsk( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    __shared__ RealSample sumShmem[ APSM_BLOCK_BATCH_SIZE ];
    sumShmem[ blockThreadId ] = 0;

    for ( uint32_t idx = blockThreadId; idx < numElements / 2; idx += batchSize )
    {
        bool symbolError = signbit( y_true[ 2 * idx ] ) != signbit( y_hat[ 2 * idx ] );
        sumShmem[ blockThreadId ] += symbolError;
    }

    sumResults( sumShmem, serReturn, blockThreadId );
}

RealSample NomaTrainHistory::computeSer( const RealSample* y_true, const RealSample* y_hat, uint32_t numElements )
{
    if ( bitPerSymbol == 1 )
    {
        kernelComputeSerBpsk<<<1, APSM_BLOCK_BATCH_SIZE, 0, stream>>>( y_true, y_hat, numElements );
    }
    else
    {
        const float maxConstellationVal = ( ( 1 << ( bitPerSymbol / 2 ) ) - 1 ) * modulationScale;

        kernelComputeSer<<<1, APSM_BLOCK_BATCH_SIZE, 0, stream>>>( y_true, y_hat, numElements, modulationScale, maxConstellationVal );
    }

    return getResult( &serReturn ) / ( numElements / 2 );
}

__device__ uint32_t demap1D( const RealSample x, const uint32_t num_symbols )
{
    // convert ... -7 -5 -3 -1 1 3 5 7 ... to 0 1 2 3 4 5 6 7 ...
    int32_t xi = (int32_t)round( ( x + num_symbols - 1 ) / 2 );

    // limit to the valid range
    xi = min( (int32_t)num_symbols - 1, max( 0, xi ) );

    const uint32_t xi_gc = xi ^ ( xi >> 1 );
    return xi_gc;
}

__device__ uint32_t bitErrors1D( const RealSample y_true, const RealSample y_hat, const uint32_t numSymbols, const RealSample revScale )
{
    uint32_t bits_true = demap1D( y_true * revScale, numSymbols );
    uint32_t bits_hat = demap1D( y_hat * revScale, numSymbols );

    return __popc( bits_true ^ bits_hat );
}

__device__ RealSample berReturn;

__global__ void kernelComputeBer( const RealSample* y_true, const RealSample* y_hat, const uint32_t numElements, const uint32_t bps, const RealSample revScale )
{
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    __shared__ RealSample sumShmem[ APSM_BLOCK_BATCH_SIZE ];
    sumShmem[ blockThreadId ] = 0;

    const uint32_t numSymbols = bps == 1 ? 2 : 1 << ( bps / 2 );

    for ( uint32_t idx = blockThreadId; idx < numElements / 2; idx += batchSize )
    {
        sumShmem[ blockThreadId ] += bitErrors1D( y_true[ 2 * idx ], y_hat[ 2 * idx ], numSymbols, revScale );
        if ( bps > 1 )
            sumShmem[ blockThreadId ] += bitErrors1D( y_true[ 2 * idx + 1 ], y_hat[ 2 * idx + 1 ], numSymbols, revScale );
    }

    sumResults( sumShmem, berReturn, blockThreadId );
}

RealSample NomaTrainHistory::computeBer( const RealSample* y_true, const RealSample* y_hat, const uint32_t numElements )
{
    kernelComputeBer<<<1, APSM_BLOCK_BATCH_SIZE, 0, stream>>>( y_true, y_hat, numElements, bitPerSymbol, 1. / modulationScale );
    return getResult( &berReturn ) / ( numElements * bitPerSymbol / 2 );
}

void NomaTrainHistory::toCsv( std::string filename )
{
    assert( evmTrain.size() == evmVal.size() );

    std::ofstream file;
    file.open( filename, std::ios::out | std::ios::trunc );
    file << "step,train_evm,train_ser,train_ber,val_evm,val_ser,val_ber" << std::endl;
    for ( uint32_t idx = 0; idx < evmTrain.size(); idx++ )
    {
        file << idx << "," << evmTrain[ idx ] << "," << serTrain[ idx ] << "," << berTrain[ idx ] << ",";
        file << evmVal[ idx ] << "," << serVal[ idx ] << "," << berVal[ idx ] << std::endl;
    }
    file.close();
}
