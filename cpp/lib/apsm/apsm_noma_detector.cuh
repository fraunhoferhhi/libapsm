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
 * @file apsm_noma_detector.cuh
 * @brief APSM NOMA detector header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.05.27   0.01    initial version
 */

// include guard
#pragma once

// STD C
#include <memory>
#include <utility>

// CUDA
#include <cuda_runtime.h>

// JSON
#include <nlohmann/json.hpp>

// APSM
#include <apsm/apsm_parameters.cuh>
#include <cuda/cuda_stream.cuh>
#include <noma/noma_detector.cuh>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// Forward declare cuda matrix classes (cuda_matrix.cuh can't be included here, because when thrust headers are included from gcc you will get some completely illegible error messages)
class HostTrainingState;
class HostStatisticState;
class CudaHostDedupMatrix;
class CudaHostMatrix;

template <int32_t version_id>
class ApsmNomaDetector : public NomaDetector
{
public:
    ApsmNomaDetector( const apsm_parameters& par, uint32_t numAntennas );
    ~ApsmNomaDetector();
    virtual float train( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining, NomaValSet valSet = std::nullopt );
    virtual float detect( const ComplexSampleMatrix& rxSigData, ComplexSampleVector& estData );
    virtual void resetState();

    RealSample computeNormRatio();
    HostTrainingState* getTrainingState() { return trainingState.get(); }

private:
    float wrapperTrain( const CudaHostDedupMatrix& d_apsm_rxd2r, const CudaHostMatrix& d_apsm_txd1r, std::optional<std::tuple<NomaTrainHistory&, const CudaHostDedupMatrix, const CudaHostMatrix>> valSet );
    float wrapperDetect( const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r );
    void wrapperAdaptCoeffs();

    RealSample optimizeGaussianVariance( const ComplexSampleMatrix&, const ComplexSampleVector& );
    RealSample computeGaussianVarianceHeuristic( const ComplexSampleMatrix& );
    void llsInitialization( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleVector& txSigTraining );

    apsm_parameters par;
    uint32_t numAntennas;
    std::unique_ptr<HostTrainingState> trainingState;
    CudaStream stream;
    RealSample* deviceBuffer;
};

class ApsmNomaDetectorFactory
{
public:
    static std::unique_ptr<NomaDetector> build( const nlohmann::json& config, uint32_t numAntennas );
};

/**
 * @}
 */
