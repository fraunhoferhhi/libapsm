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
 * @file lls_detector.cu
 * @brief linear least squared detector
 *
 * @author Danie Schäufele    HHI,
 *
 * @date 2021.10.xx   0.01    initial version
 */

// STD C
#include <iostream>

// APSM
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_eventtimer.cuh"
#include "cuda/cuda_indexing.cuh"
#include "cuda/cuda_matrix_device.cuh"
#include "cuda/cuda_matrix_host.cuh"
#include "noma/lls_detector.cuh"

#define CUBLAS_CHECK_THROW( a )                                               \
    {                                                                         \
        cublasStatus_t res = a;                                               \
        if ( res == CUBLAS_STATUS_NOT_INITIALIZED )                           \
            throw std::runtime_error( "CUBLAS: Not initialized (properly)" ); \
        else if ( res == CUBLAS_STATUS_ALLOC_FAILED )                         \
            throw std::runtime_error( "CUBLAS: Allocation failed" );          \
        else if ( res == CUBLAS_STATUS_INVALID_VALUE )                        \
            throw std::runtime_error( "CUBLAS: Invalid value" );              \
        else if ( res == CUBLAS_STATUS_ARCH_MISMATCH )                        \
            throw std::runtime_error( "CUBLAS: Architecture mismatch" );      \
        else if ( res == CUBLAS_STATUS_MAPPING_ERROR )                        \
            throw std::runtime_error( "CUBLAS: Mapping error" );              \
        else if ( res == CUBLAS_STATUS_EXECUTION_FAILED )                     \
            throw std::runtime_error( "CUBLAS: Execution failed" );           \
        else if ( res == CUBLAS_STATUS_INTERNAL_ERROR )                       \
            throw std::runtime_error( "CUBLAS: Internal error" );             \
        else if ( res == CUBLAS_STATUS_NOT_SUPPORTED )                        \
            throw std::runtime_error( "CUBLAS: Not supported" );              \
        else if ( res == CUBLAS_STATUS_LICENSE_ERROR )                        \
            throw std::runtime_error( "CUBLAS: License error" );              \
        else if ( res != CUBLAS_STATUS_SUCCESS )                              \
            throw std::runtime_error( "CUBLAS: Unknown error" );              \
    }

// debug function for printing a matrix in cuBLAS format
void printMat( float* mat, uint32_t rows, uint32_t cols )
{
    float* data = (float*)malloc( rows * cols * sizeof( float ) );
    CUDA_CHECK( cudaMemcpy( data, mat, rows * cols * sizeof( float ), cudaMemcpyDeviceToHost ) );
    std::cout << "[ " << std::endl;
    for ( uint32_t row = 0; row < rows; row++ )
    {
        std::cout << "[ ";
        for ( uint32_t col = 0; col < cols; col++ )
        {
            std::cout << data[ IDX2E( row, col, rows ) ] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]" << std::endl;
    free( data );
}

LlsNomaDetector::LlsNomaDetector( uint32_t numAntennas, std::shared_ptr<CudaStream> extStream )
{
    if ( extStream )
        stream = extStream;
    else
        stream = std::make_shared<CudaStream>();
    CUBLAS_CHECK_THROW( cublasCreate( &cublasHandle ) );
    CUBLAS_CHECK_THROW( cublasSetStream( cublasHandle, *stream ) );
    CUDA_CHECK( cudaMalloc( &weightMat, 2 * numAntennas * 2 * sizeof( float ) ) );
    CUDA_CHECK( cudaMalloc( &inputPointerMem, sizeof( float* ) ) );
    CUDA_CHECK( cudaMalloc( &outputPointerMem, sizeof( float* ) ) );
}

LlsNomaDetector::~LlsNomaDetector()
{
    CUDA_CHECK( cudaFree( weightMat ) );
    CUDA_CHECK( cudaFree( inputPointerMem ) );
    CUDA_CHECK( cudaFree( outputPointerMem ) );
    try
    {
        CUBLAS_CHECK_THROW( cublasDestroy( cublasHandle ) );
    }
    catch ( std::runtime_error& err )
    {
        std::cerr << err.what() << std::endl;
        exit( EXIT_FAILURE );
    }
}

float LlsNomaDetector::train( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining, NomaValSet valSet )
{
    const int32_t numOutputDims = 1;
    int32_t numSamples = rxSigTraining[ 0 ].size() * 2;
    int32_t numInputDims = rxSigTraining.size() * 2;

    CudaHostDedupMatrix d_rxSigTraining( *stream, rxSigTraining );
    CudaHostMatrix d_txSigTraining( *stream, txSigTraining );
    RealSample* rxSigMat = d_rxSigTraining.allocateAndCopyToMemory( *stream );
    RealSample* txSigMat = d_txSigTraining.allocateAndCopyToMemory( *stream );

    // Initialise timer
    CUDA_EventTimer timer;

    timer.start( *stream );
    trainLowLevel( rxSigMat, txSigMat, numInputDims, numOutputDims, numSamples );
    timer.stop( *stream );

    // sync after copying
    stream->synchronize();

    CUDA_CHECK( cudaFree( rxSigMat ) );
    CUDA_CHECK( cudaFree( txSigMat ) );

    return timer.elapsed();
}

void LlsNomaDetector::trainLowLevel( const RealSample* rxSigMat, const RealSample* txSigMat, const int32_t numInputDims, const int32_t numOutputDims, const int32_t numSamples )
{
    wrapPointerInArray( rxSigMat, inputPointerMem );
    wrapPointerInArray( txSigMat, outputPointerMem );
    const int32_t m = numSamples;
    const int32_t n = numInputDims;
    const int32_t k = numOutputDims;
    const int32_t ldInput = numSamples;
    const int32_t ldOutput = numSamples;
    int32_t info = 0;

    // run actual algorithm to find least squares solution
    CUBLAS_CHECK_THROW( cublasSgelsBatched( cublasHandle, CUBLAS_OP_N, m, n, k, inputPointerMem, ldInput, outputPointerMem, ldOutput, &info, nullptr, 1 ) );

    // copy weight matrix to member variable
    CUDA_CHECK( cudaMemcpyAsync( weightMat, txSigMat, numInputDims * sizeof( RealSample ), cudaMemcpyDeviceToDevice, *stream ) );
}

float LlsNomaDetector::detect( const ComplexSampleMatrix& rxSigData, ComplexSampleVector& estData )
{
    const int32_t numOutputDims = 1;
    int32_t numSamples = rxSigData[ 0 ].size() * 2;
    int32_t numInputDims = rxSigData.size() * 2;

    CudaHostDedupMatrix d_rxSigData( *stream, rxSigData );
    RealSample* rxSigMat = d_rxSigData.allocateAndCopyToMemory( *stream );
    RealSample* estMat;
    CUDA_CHECK( cudaMalloc( &estMat, numSamples * numOutputDims * sizeof( RealSample ) ) );

    // Initialise timer
    CUDA_EventTimer timer;

    timer.start( *stream );
    detectLowLevel( rxSigMat, estMat, numInputDims, numOutputDims, numSamples );
    timer.stop( *stream );

    copyResult( estMat, estData, numSamples / 2 );

    CUDA_CHECK( cudaFree( rxSigMat ) );
    CUDA_CHECK( cudaFree( estMat ) );

    return timer.elapsed();
}

void LlsNomaDetector::detectLowLevel( const RealSample* rxSigMat, RealSample* estMat, const int32_t numInputDims, const int32_t numOutputDims, const int32_t numSamples, const float alpha, const float beta )
{
    uint32_t m = numSamples;
    uint32_t n = numOutputDims;
    uint32_t k = numInputDims;
    uint32_t lda = m;
    uint32_t ldb = k;
    uint32_t ldc = m;

    // run matrix multiplication ( extMat = rxSigmat * weightMat )
    CUBLAS_CHECK_THROW( cublasSgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, rxSigMat, lda, weightMat, ldb, &beta, estMat, ldc ) );
}

void LlsNomaDetector::resetState()
{
    // do nothing because state is completely replaced by train anyway
}

void LlsNomaDetector::copyResult( RealSample* src, ComplexSampleVector& m, uint32_t numComplexValues )
{
    m.resize( numComplexValues );
    CUDA_CHECK( cudaMemcpyAsync( m.data(), src, numComplexValues * 2 * sizeof( RealSample ), cudaMemcpyDeviceToHost, *stream ) );
}

void LlsNomaDetector::wrapPointerInArray( const float* ptr, float** pointerMem )
{
    const float* const h_array[] = { ptr };
    CUDA_CHECK( cudaMemcpyAsync( pointerMem, h_array, sizeof( float* ), cudaMemcpyHostToDevice, *stream ) );
}

float LlsNomaDetector::computeSignalPower( const RealSample* data, const uint32_t numValues )
{
    float result;
    CUBLAS_CHECK_THROW( cublasSdot( cublasHandle, numValues, data, 1, data, 1, &result ) );
    stream->synchronize();
    return sqrtf( result / numValues );
}

std::unique_ptr<NomaDetector> LlsNomaDetectorFactory::build( const nlohmann::json& config, uint32_t numAntennas )
{
    auto validKeys = { "otype" };
    for ( auto& [ key, value ] : config.items() )
    {
        if ( std::find( validKeys.begin(), validKeys.end(), key ) == validKeys.end() )
            throw std::invalid_argument( "Invalid key: " + key );
    }

    return std::make_unique<LlsNomaDetector>( numAntennas );
}
