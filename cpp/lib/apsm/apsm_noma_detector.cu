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
 * @file apsm_noma_detector.cu
 * @brief APSM NOMA detector
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.05.27   0.01    initial version
 */

// STD C
#include <algorithm>
#include <bitset>

// CUDA
#include <cooperative_groups.h>

// APSM
#include "apsm/apsm_detect.cuh"
#include "apsm/apsm_noma_detector.cuh"
#include "apsm/apsm_versions.cuh"
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_indexing.cuh"
#include "cuda/cuda_matrix_host.cuh"
#include "noma/lls_detector.cuh"

// set namespace for cooperative groups
namespace cg = cooperative_groups;

template <int32_t version_id>
ApsmNomaDetector<version_id>::ApsmNomaDetector( const apsm_parameters& par, uint32_t numAntennas )
    : par( par )
    , numAntennas( numAntennas )
{
    CUDA_CHECK( cudaMalloc( &deviceBuffer, sizeof( RealSample ) * 2 * numAntennas * 2 * par.dictionarySize ) );
    resetState();

    if ( par.gaussianKernelVariance != 0 )
        this->par.gaussianKernelVariance *= numAntennas * numAntennas;
}

template <int32_t version_id>
ApsmNomaDetector<version_id>::~ApsmNomaDetector()
{
    CUDA_CHECK( cudaFree( deviceBuffer ) );
}

template <int32_t version_id>
void ApsmNomaDetector<version_id>::resetState()
{
    trainingState.reset( new HostTrainingState( 2 * numAntennas, 2 * par.dictionarySize ) );
}

template <int32_t version_id>
float ApsmNomaDetector<version_id>::train( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining, NomaValSet valSet )
{
    if ( par.llsInitialization )
        llsInitialization( rxSigTraining, txSigTraining );

    ComplexSampleMatrix rxSigTrainingShuffled = rxSigTraining;
    ComplexSampleVector txSigTrainingShuffled = txSigTraining;

    if ( par.trainingShuffle )
        shuffleData( rxSigTrainingShuffled, txSigTrainingShuffled );

    CudaHostDedupMatrix d_apsm_rxd2r( stream, rxSigTrainingShuffled );
    CudaHostMatrix d_apsm_txd1r( stream, txSigTrainingShuffled );

    if ( par.gaussianKernelVariance == 0 )
        // par.gaussianKernelVariance = optimizeGaussianVariance( rxSigTraining, txSigTraining );
        par.gaussianKernelVariance = computeGaussianVarianceHeuristic( rxSigTrainingShuffled );

    std::optional<std::tuple<NomaTrainHistory&, const CudaHostDedupMatrix, const CudaHostMatrix>> wrapperValSet = std::nullopt;
    if ( valSet.has_value() )
    {
        auto [ valStats, rxSigVal, txSigVal ] = *valSet;
        CudaHostDedupMatrix d_rxSigVal( stream, rxSigVal );
        CudaHostMatrix d_txSigVal( stream, txSigVal );
        wrapperValSet.emplace( std::tie(
            valStats,
            d_rxSigVal,
            d_txSigVal ) );
    }

    float trainTime = wrapperTrain( d_apsm_rxd2r, d_apsm_txd1r, wrapperValSet );
    return trainTime;
}

template <int32_t version_id>
float ApsmNomaDetector<version_id>::detect( const ComplexSampleMatrix& rxSigData, ComplexSampleVector& estData )
{
    CudaHostDedupMatrix d_apsm_ryd2r( stream, rxSigData );
    CudaHostMatrix d_apsm_esd2r( 1, d_apsm_ryd2r.getWidth() );

    float detectTime = wrapperDetect( d_apsm_ryd2r, d_apsm_esd2r );

    ThrustComplexSampleDeviceVector d_esdat_vec = d_apsm_esd2r.toComplexVector( stream );
    estData.resize( d_esdat_vec.size() );
    thrust::copy( d_esdat_vec.begin(), d_esdat_vec.end(), estData.begin() );

    return detectTime;
}

template <int32_t version_id>
RealSample ApsmNomaDetector<version_id>::optimizeGaussianVariance( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining )
{
    RealSample origGaussianVariance = par.gaussianKernelVariance; // the parameters get modified by this function and restored at the end
    const uint32_t numTrainSamples = rxSigTraining[ 0 ].size() * 0.8;

    ComplexSampleMatrix rxTrainSet, rxValSet;
    ComplexSampleVector txTrainSet, txValSet;
    rxTrainSet.reserve( rxSigTraining.size() );
    rxValSet.reserve( rxSigTraining.size() );

    for ( uint32_t idx = 0; idx < rxSigTraining.size(); idx++ )
    {
        rxTrainSet.push_back( ComplexSampleVector( rxSigTraining[ idx ].begin(), rxSigTraining[ idx ].begin() + numTrainSamples ) );
        rxValSet.push_back( ComplexSampleVector( rxSigTraining[ idx ].begin() + numTrainSamples, rxSigTraining[ idx ].end() ) );
    }
    txTrainSet = ComplexSampleVector( txSigTraining.begin(), txSigTraining.begin() + numTrainSamples );
    txValSet = ComplexSampleVector( txSigTraining.begin() + numTrainSamples, txSigTraining.end() );

    const RealSample gaussianVariances[] = { 0.001, 0.01, 0.1, 1, 10 };
    RealSample bestGv = 0;
    RealSample bestEvm = std::numeric_limits<RealSample>::infinity();
    for ( RealSample gv : gaussianVariances )
    {
        ComplexSampleVector estVal;
        par.gaussianKernelVariance = gv;
        train( rxTrainSet, txTrainSet );
        detect( rxValSet, estVal );

        RealSample squaredSum = 0;
        for ( uint32_t idx = 0; idx < estVal.size(); idx++ )
        {
            float error = std::abs( estVal[ idx ] - txValSet[ idx ] );
            squaredSum += error * error;
        }

        if ( squaredSum < bestEvm )
        {
            bestEvm = squaredSum;
            bestGv = gv;
        }
    }

    std::cout << "Best Gaussian variance = " << bestGv << std::endl;
    par.gaussianKernelVariance = origGaussianVariance;
    return bestGv;
}

template <int32_t version_id>
RealSample ApsmNomaDetector<version_id>::computeGaussianVarianceHeuristic( const ComplexSampleMatrix& rxSigTraining )
{
    RealSampleVector normDistances;
    for ( uint32_t idx = 0; idx < rxSigTraining[ 0 ].size(); idx++ )
    {
        for ( uint32_t jdx = idx + 1; jdx < rxSigTraining[ 0 ].size(); jdx++ )
        {
            RealSample dist = 0;
            for ( uint32_t kdx = 0; kdx < rxSigTraining.size(); kdx++ )
            {
                ComplexSample diff = rxSigTraining[ kdx ][ idx ] - rxSigTraining[ kdx ][ jdx ];
                dist += std::abs( diff );
            }
            normDistances.push_back( std::sqrt( dist ) );
        }
    }
    std::sort( normDistances.begin(), normDistances.end() );

    const RealSample p = 0.7;
    const RealSample q = 5;
    RealSample variance = normDistances[ normDistances.size() * p ] * std::pow( rxSigTraining[ 0 ].size(), -1. / q );
    variance = variance * variance;
    std::cout << "Gaussian variance heuristic = " << variance << std::endl;
    return variance;
}

__global__ void kernel_norm_ratio( DeviceTrainingState trainingState, const apsm_parameters par, RealSample* reduction_memory )
{
    // copied from training -> kernel_adapt_coeffs
    // TODO: refactor
    cg::grid_group grid = cg::this_grid();

    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();

    const uint32_t batch_size = getBlockDim_1D();

    extern __shared__ RealSample shared[];
    shared[ blockThreadId ] = 0;

    for ( uint32_t basis_idx = blockThreadId; basis_idx < trainingState.basis.getUsedHeight(); basis_idx += batch_size )
    {
        RealSample firstCoeff = trainingState.gaussianCoeffs( blockId );
        RealSample secondCoeff = trainingState.gaussianCoeffs( basis_idx );

        // calculate weight
        RealSample exp_weight = RealSample( -0.5 ) / par.gaussianKernelVariance;

        // calculate argument
        RealSample exp_argument = RealSample( 0.0 );

#pragma unroll
        for ( uint32_t dim = 0; dim < trainingState.basis.getWidth(); dim++ )
        {
            RealSample dist_element = trainingState.basis( blockId, dim ) - trainingState.basis( basis_idx, dim );
            exp_argument += dist_element * dist_element;
        }

        shared[ blockThreadId ] += exp( exp_weight * exp_argument ) * firstCoeff * secondCoeff;
    }

    __syncthreads();

    if ( blockThreadId == 0 )
    {
        RealSample result = 0;
        for ( uint32_t idx = 0; idx < min( trainingState.basis.getUsedHeight(), batch_size ); idx++ )
        {
            result += shared[ idx ];
        }
        reduction_memory[ blockId ] = result;
    }

    cg::sync( grid ); // Sync whole grid

    if ( blockId == 0 && blockThreadId == 0 )
    {
        RealSample gaussian_norm = 0;
        for ( uint32_t idx = 0; idx < trainingState.basis.getUsedHeight(); idx++ )
        {
            gaussian_norm += reduction_memory[ idx ];
        }
        gaussian_norm *= par.gaussianKernelWeight;

        RealSample linear_norm = 0;
        for ( uint32_t idx = 0; idx < trainingState.linearCoeffs.getHeight(); idx++ )
        {
            linear_norm += trainingState.linearCoeffs( idx ) * trainingState.linearCoeffs( idx );
        }
        linear_norm *= par.linearKernelWeight;

        reduction_memory[ 0 ] = linear_norm;
        reduction_memory[ 1 ] = gaussian_norm;
    }
}

template <int32_t version_id>
RealSample ApsmNomaDetector<version_id>::computeNormRatio()
{
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, trainingState->basis.getUsedHeight() );

    uint32_t sharedMemorySize = trainingState->basis.getUsedHeight() * sizeof( RealSample );

    RealSample* reduction_memory;
    CUDA_CHECK( cudaMalloc( &reduction_memory, sizeof( RealSample ) * trainingState->basis.getUsedHeight() ) );

    DeviceTrainingState trainingState_device = trainingState->toDevice();

    void* kernelArgs[] = {
        (void*)&trainingState_device,
        (void*)&par,
        (void*)&reduction_memory
    };

    CUDA_CHECK( cudaLaunchCooperativeKernel( (void*)kernel_norm_ratio,
                                             grid_dim, block_dim, kernelArgs,
                                             sharedMemorySize, stream ) );

    stream.synchronize();

    RealSample h_results[ 2 ];
    CUDA_CHECK( cudaMemcpy( h_results, reduction_memory, sizeof( RealSample ) * 2, cudaMemcpyDeviceToHost ) );

    return h_results[ 0 ] / h_results[ 1 ];
}

__global__ void copyWeightMat( DeviceTrainingState trainingState, float* weightMat, const apsm_parameters par )
{
    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    for ( uint32_t idx = blockThreadId; idx < trainingState.linearCoeffs.getHeight(); idx += batchSize )
        trainingState.linearCoeffs( idx ) = weightMat[ idx ] / par.linearKernelWeight;
}

template <int32_t version_id>
void ApsmNomaDetector<version_id>::llsInitialization( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining )
{
    LlsNomaDetector lls( numAntennas );
    lls.train( rxSigTraining, txSigTraining );
    float* weightMat = lls.getWeightMat();

    DeviceTrainingState trainingState_device = trainingState->toDevice();

    dim3 blockDim, gridDim;
    apsm_kernel_dims( &blockDim, &gridDim, 1 );

    copyWeightMat<<<gridDim, blockDim, 0, stream>>>( trainingState_device, weightMat, par );

    stream.synchronize();
}

std::unique_ptr<NomaDetector> ApsmNomaDetectorFactory::build( const nlohmann::json& config, uint32_t numAntennas )
{
    auto validKeys = { "otype", "llsInitialization",
                       "trainingShuffle", "gaussianKernelWeight", "gaussianKernelVariance", "windowSize",
                       "sampleStep", "trainPasses", "startWithFullWindow", "dictionarySize", "normConstraint",
                       "eB", "detectVersion" };
    for ( auto& [ key, value ] : config.items() )
    {
        if ( std::find( validKeys.begin(), validKeys.end(), key ) == validKeys.end() )
            throw std::invalid_argument( "Invalid key: " + key );
    }

    apsm_parameters par;
    par.llsInitialization = config.value( "llsInitialization", false );
    par.trainingShuffle = config.value( "trainingShuffle", false );
    par.gaussianKernelWeight = config.value( "gaussianKernelWeight", 0.5 );
    par.linearKernelWeight = 1 - par.gaussianKernelWeight;
    par.gaussianKernelVariance = config.value( "gaussianKernelVariance", 0.05 );
    par.windowSize = config.value( "windowSize", 20 );
    par.sampleStep = config.value( "sampleStep", 2 );
    par.trainPasses = config.value( "trainPasses", 1 );
    par.startWithFullWindow = config.value( "startWithFullWindow", false );
    par.dictionarySize = config.value( "dictionarySize", 685 );
    par.normConstraint = config.value( "normConstraint", 0. );
    par.eB = config.value( "eB", 0.001 );
    if ( config.value( "detectVersion", "balanced" ) == "original" )
        return std::make_unique<ApsmNomaDetector<apsm_versions::APSM_DETECT_ORIGINAL>>( par, numAntennas );
    else if ( config.value( "detectVersion", "balanced" ) == "oldfast" )
        return std::make_unique<ApsmNomaDetector<apsm_versions::APSM_DETECT_OLDFAST>>( par, numAntennas );
    else if ( config.value( "detectVersion", "balanced" ) == "shmem" )
        return std::make_unique<ApsmNomaDetector<apsm_versions::APSM_DETECT_SHMEM>>( par, numAntennas );
    else if ( config.value( "detectVersion", "balanced" ) == "balanced" )
        return std::make_unique<ApsmNomaDetector<apsm_versions::APSM_DETECT_BALANCED>>( par, numAntennas );
    else
        throw std::invalid_argument( "Unknown detect version: " + config.value( "detectVersion", "balanced" ) );
}

// explicit instantiation of all known versions of the wrapper, so they are present in the library
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_ORIGINAL>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_OLDFAST>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_SHMEM>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_BALANCED>;
