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
 * @file apsm_train.cu
 * @brief APSM train kernel
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.12.10   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// CUDA
#include <cooperative_groups.h>

// APSM
#include "apsm/apsm_noma_detector.cuh"
#include "apsm/apsm_train.cuh"
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_eventtimer.cuh"
#include "cuda/cuda_indexing.cuh"

// set namespace for cooperative groups
namespace cg = cooperative_groups;

//#define INPUT_SHAREDMEM

/**
 * @brief C++ wrapper function for APSM train part.
 * @details This function call the APSM CUDA train kernel.
 *
 * @param[out] d_apsm_basis basis matrix (learned dictionary)
 * @param[out] d_apsm_coeff coefficient vector (learned dictionary)
 * @param[in] d_apsm_rxd2r received rx constallation vector
 * @param[in] d_apsm_txd1r transmitted tx constallation vector (pilots)
 * @param[in] par APSM parameter set
 *
 * @return measured kernel processing time
 */
template <int32_t version_id>
float ApsmNomaDetector<version_id>::wrapperTrain( const CudaHostDedupMatrix& d_apsm_rxd2r, const CudaHostMatrix& d_apsm_txd1r, std::optional<std::tuple<NomaTrainHistory&, const CudaHostDedupMatrix, const CudaHostMatrix>> valSet )
{

    // TODO: (mm) we have to check this assertions
    // Guilermo: Please double check the assertions here.
    //           There are 3 assertions that need to be checked for correctness when/if parameters can change.
    assert( par.windowSize <= par.dictionarySize ); // window can't be larger than dictionary
    assert( par.trainPasses == 1 || par.dictionarySize * 2 == d_apsm_rxd2r.getWidth() ); // dictionary sparsification + multiple train passes doesn't make sense

    // Initialise timer
    CUDA_EventTimer timer;

    trainingState->basis.pushRowsFromMatrix( stream, d_apsm_rxd2r );

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, 2 );

    timer.start( stream );

    if ( par.startWithFullWindow )
    {
        trainingState->basis.moveWindow( 2 * par.windowSize - par.sampleStep );
        trainingState->gaussianCoeffs.moveWindow( 2 * par.windowSize - par.sampleStep );
    }

    for ( uint32_t idx = 0; idx < par.trainPasses; idx++ )
    {
        for ( uint32_t sample_idx = par.startWithFullWindow ? 2 * par.windowSize : par.sampleStep;
              sample_idx <= d_apsm_rxd2r.getWidth(); sample_idx += par.sampleStep )
        {
            if ( idx == 0 )
            {
                trainingState->basis.moveWindow( par.sampleStep );
                trainingState->gaussianCoeffs.moveWindow( par.sampleStep );
            }

            uint32_t windowSize = min( 2 * par.windowSize, sample_idx );
            grid_dim.x = windowSize;

            // calculate shared memory size parameter
            uint32_t sharedMemorySize = 0;
            sharedMemorySize += APSM_BLOCK_BATCH_SIZE * sizeof( RealSample );

            DeviceTrainingState trainingState_device = trainingState->toDevice();
            CudaDeviceDedupMatrix d_apsm_rxd2r_device = d_apsm_rxd2r.toDevice();
            CudaDeviceMatrix d_apsm_txd1r_device = d_apsm_txd1r.toDevice();

            void* kernelArgs[] = {
                (void*)&trainingState_device,
                (void*)&d_apsm_rxd2r_device,
                (void*)&d_apsm_txd1r_device,
                (void*)&par,
                (void*)&deviceBuffer,
                (void*)&sample_idx,
            };

            CUDA_CHECK( cudaLaunchCooperativeKernel( (void*)kernel_apsm_train,
                                                     grid_dim, block_dim, kernelArgs,
                                                     sharedMemorySize, stream ) );

            if ( par.normConstraint > 0 )
            {
                wrapperAdaptCoeffs();
            }

            if ( valSet.has_value() )
            {
                auto [ valStats, d_rxSigVal, d_txSigVal ] = *valSet;

                CudaHostMatrix d_estDataTrain( 1, d_apsm_rxd2r.getWidth() );
                wrapperDetect( d_apsm_rxd2r, d_estDataTrain );
                valStats.addTrainStep( d_apsm_txd1r.getRawDataPointer(), d_estDataTrain.getRawDataPointer(), d_apsm_txd1r.getWidth() );

                CudaHostMatrix d_estDataVal( 1, d_rxSigVal.getWidth() );
                wrapperDetect( d_rxSigVal, d_estDataVal );
                valStats.addValStep( d_txSigVal.getRawDataPointer(), d_estDataVal.getRawDataPointer(), d_txSigVal.getWidth() );
            }
        }
    }

    // stop timer
    timer.stop( stream );

    return timer.elapsed();
}

template <int32_t version_id>
void ApsmNomaDetector<version_id>::wrapperAdaptCoeffs()
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

    CUDA_CHECK( cudaLaunchCooperativeKernel( (void*)kernel_adapt_coeffs,
                                             grid_dim, block_dim, kernelArgs,
                                             sharedMemorySize, stream ) );
}

__global__ void kernel_adapt_coeffs( DeviceTrainingState trainingState, const apsm_parameters par, RealSample* reduction_memory )
{
    cg::grid_group grid = cg::this_grid();

    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();

    const uint32_t batch_size = getBlockDim_1D();

    extern __shared__ RealSample shared[];
    shared[ blockThreadId ] = 0;

    for ( uint32_t basis_idx = blockThreadId; basis_idx < trainingState.basis.getUsedHeight(); basis_idx += batch_size )
    {
        // TODO: use more flexible method for passing basis vector
        RealSample secondBasisVector[ 32 ]; // TODO: (mm) Where is this 32 is comming from ???
        for ( uint32_t sample_idx = 0; sample_idx < trainingState.basis.getWidth(); sample_idx++ )
            secondBasisVector[ sample_idx ] = trainingState.basis( basis_idx, sample_idx );
        RealSample firstCoeff = trainingState.gaussianCoeffs( blockId );
        RealSample secondCoeff = trainingState.gaussianCoeffs( basis_idx );

        // TODO: fix second basis vector
        shared[ blockThreadId ] += par.gaussianKernelWeight * kernel_apsm_detection_gaussian_original( trainingState.basis.getWidth(), trainingState.basis, blockId, secondBasisVector, par.gaussianKernelVariance )
            * firstCoeff * secondCoeff;
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
        RealSample result = 0;
        for ( uint32_t idx = 0; idx < trainingState.basis.getUsedHeight(); idx++ )
        {
            result += reduction_memory[ idx ];
        }
        if ( result > 0 )
            result = sqrt( result );
        reduction_memory[ 0 ] = result;
        // printf("Correction result = %f\n", reduction_memory[0]);
    }

    cg::sync( grid ); // Sync whole grid

    if ( blockThreadId == 0 )
    {
        if ( reduction_memory[ 0 ] > par.normConstraint )
        {
            RealSample correction_factor = par.normConstraint / reduction_memory[ 0 ];

            trainingState.gaussianCoeffs( blockId, 0 ) *= correction_factor;
        }
    }
}

/**
 * @brief CUDA train kernel
 * @details
 *
 * @param[out] basis
 * @param[out] coeff
 * @param[in] rx_data
 * @param[in] train_data
 * @param[in] par
 * @param[in] d_rxdata_input
 *
 * @return void
 */
__global__ void kernel_apsm_train( DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, const CudaDeviceMatrix train_data, const apsm_parameters par, RealSample* d_rxdata_input )
{

    // to sync across the whole grid
    // cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    // register for unique indexing
    // const uint32_t threadId	    = getThreadIdx_1D_1D();
    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();

    const uint32_t batch_size = getBlockDim_1D();

    const uint32_t linearLength = rx_data.getHeight();
    // const uint32_t numberOfSamples = rx_data.width;

    const uint32_t gaussianLength = trainingState.basis.getUsedHeight();

    extern __shared__ RealSample shared[];

    // Shared memory for detected symbol vector
    RealSample* detectedSymbol = &shared[ 0 ]; // detectedSymbol is manually set at the beginning of shared mem
    //	RealSample *rxdata_input = &shared[1];	// shared memory for rxinput_data

    // ---------------------------------------------------------------------------

    // If thread index is out of bounds, do nothing.
    if ( blockId >= 2 * par.windowSize )
        return;

        // copy input data to shared memory
#ifdef INPUT_SHAREDMEM
    RealSample* rxdata_input_global = d_rxdata_input + ( linearLength * blockId );
    RealSample* rxdata_input = &shared[ batch_size ];

    for ( uint32_t idx = 0; idx < linearLength; idx += batch_size )
        rxdata_input[ idx + blockThreadId ] = rxdata_input_global[ idx + blockThreadId ];
    __syncthreads();
#else
    RealSample* rxdata_input = d_rxdata_input + ( linearLength * blockId );
#endif

#if 0
    if ( ( blockThreadId == 0 ) && ( blockId == 0 ) )
    {
        printf( "rxdata_input [size=%d,blockId=%d]: ", linearLength, blockId );
        for ( uint32_t idx = 0; idx < linearLength; idx++ )
            printf( "%f ", rxdata_input[ idx ] );
        printf( "\n" );
    }
    __syncthreads();
#endif

    // ---------------------------------------------------------------------------

    // Next step is to loop over all input samples, regarding to the window size
    // only par.windowsSize threads are running in each round
    {

        // determine how many and which threads are involved to calculate
        // should be 0,2,4,...,40,40,40,....40 if par.windowsSize = 20;
        uint32_t windowSize = min( 2 * par.windowSize, trainingState.basis.getUsedHeight() );

        // thread wait barrier
        //__syncthreads();
        //__threadfence();

        uint32_t threadIdStart = trainingState.basis.getUsedHeight() - windowSize;

        // only let some threads working in this round (windowing)
        if ( blockId < windowSize )
        {
            // call shared detection function
            kernel_apsm_detection_original( detectedSymbol, linearLength, gaussianLength, rxdata_input,
                                            blockId + threadIdStart, trainingState, rx_data, par );

            // write output to GPU memory  // the last thread
            if ( blockThreadId == 0 )
            {
                RealSample symbol = detectedSymbol[ 0 ];
                for ( uint32_t idx = 1; idx < batch_size; idx++ )
                    symbol += detectedSymbol[ idx ];

                detectedSymbol[ 0 ] = symbol;
            }

// write rx_data_input back to global memory
#if 0 // def INPUT_SHAREDMEM
            for ( uint32_t idx = 0; idx < linearLength; idx += batch_size )
                rxdata_input_global[ idx + blockThreadId ] = rxdata_input[ idx + blockThreadId ];
            __syncthreads();
#endif

            // we need it maybe in the future
            //__syncthreads();
            cg::sync( grid ); // Sync whole grid

            // Accumulations registers for projection
            __shared__ RealSample WContribConc;

            // compare with detected symbols with transmitted symbols
            //------------------------------------------------------------------------------------
            if ( blockThreadId == 0 )
            {
                // get tx symbol (known pilots during training phase)
                RealSample transmittedSymbol = train_data( 0, blockId + threadIdStart );

                // compute distance between tx and est. rx symbol
                RealSample symbol_distance = transmittedSymbol - detectedSymbol[ 0 ];

                // initialize with zero
                WContribConc = 0;

                if ( symbol_distance > +par.eB )
                {
                    WContribConc = symbol_distance - par.eB;
                }
                else if ( symbol_distance < -par.eB )
                {
                    WContribConc = symbol_distance + par.eB;
                }

                // calculate linear and gaussian norms
                // Because distance is zero the gaussian kernel result is always 1.0, so we can use the weight directly -> par.gaussianKernelWeight * e^( 0 )
                RealSample linearNorm = RealSample( par.linearKernelWeight ) * kernel_apsm_detection_linear_original( linearLength, rxdata_input, rxdata_input );
                RealSample gaussianNorm = RealSample( par.gaussianKernelWeight ) /* * kernel_apsm_detection_gaussian_original( linearLength, rxdata_input, rxdata_input, par.gaussianKernelVariance ) */;

                // normalization
                WContribConc /= RealSample( windowSize );
                WContribConc /= RealSample( linearNorm + gaussianNorm );

                //  Extend coefficients
                //------------------------------------------------------------------------------------

                // gaussian part
                trainingState.gaussianCoeffs( blockId + threadIdStart ) += WContribConc;
            }

            // be sure that WContribConc is visible
            __syncthreads();

            // linear part
            {
                const uint32_t batch_size = blockDim.x;

                for ( uint32_t batch_idx = 0; batch_idx < linearLength; batch_idx += batch_size )
                {

                    // read RX vector ...
                    uint32_t dim = blockThreadId + batch_idx;
                    if ( dim < linearLength )
                    {
                        // because different threads writing to the same memory address
                        atomicAdd( &trainingState.linearCoeffs( dim ), WContribConc * rxdata_input[ dim ] );
                    }
                }
            }
        }

    } // loop to calculate coeffs
}

// explicit instantiation of all known versions of the wrapper, so they are present in the library
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_ORIGINAL>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_OLDFAST>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_SHMEM>;
template class ApsmNomaDetector<apsm_versions::APSM_DETECT_BALANCED>;
