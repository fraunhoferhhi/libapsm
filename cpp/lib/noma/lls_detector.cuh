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
 * @file lls_detector.cuh
 * @brief linear least squared detector header
 *
 * @author Danie Schäufele    HHI,
 *
 * @date 2021.10.xx   0.01    initial version
 */

// include guard
#pragma once

// STD C
#include <memory>

// CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

// JSON
#include <nlohmann/json.hpp>

// APSM
#include "cuda/cuda_stream.cuh"
#include "noma/noma_detector.cuh"

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

class LlsNomaDetector : public NomaDetector
{
public:
    LlsNomaDetector( uint32_t numAntennas, std::shared_ptr<CudaStream> extStream = {} );
    ~LlsNomaDetector();

    virtual float train( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining, NomaValSet valSet = std::nullopt );
    void trainLowLevel( const RealSample* rxSigMat, const RealSample* txSigMat, const int32_t numInputDims, const int32_t numOutputDims, const int32_t numSamples );
    virtual float detect( const ComplexSampleMatrix& rxSigData, ComplexSampleVector& estData );
    void detectLowLevel( const RealSample* rxSigMat, RealSample* estMat, const int32_t numInputDims, const int32_t numOutputDims, const int32_t numSamples, const float alpha = 1, const float beta = 0 );
    virtual void resetState();
    float* getWeightMat() { return weightMat; }
    float computeSignalPower( const RealSample* data, const uint32_t numValues );

private:
    std::shared_ptr<CudaStream> stream;
    cublasHandle_t cublasHandle;
    float* weightMat;

    float** inputPointerMem;
    float** outputPointerMem;

    void copyResult( float* src, ComplexSampleVector& m, uint32_t numComplexValues );
    void wrapPointerInArray( const float* ptr, float** pointerMem );
};

class LlsNomaDetectorFactory
{
public:
    static std::unique_ptr<NomaDetector> build( const nlohmann::json& config, uint32_t numAntennas );
};

/**
 * @}
 */
