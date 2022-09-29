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
 * @file apsm_parameters.cuh
 * @brief APSM parameters header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.10.16   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// include guard
#pragma once

// APSM
#include "noma/types.cuh"

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

/**
 * @brief APSM parameters
 */
class apsm_parameters
{
public:
    // APSM kernel parameters
    RealSample linearKernelWeight; ///< Linear Kernel Weight
    RealSample gaussianKernelWeight; ///< Gaussian Kernel Weight
    RealSample gaussianKernelVariance; ///< Gaussian Kernel Variance

    // training parameters
    uint32_t windowSize; ///< number of past samples to reuse
    uint32_t sampleStep; ///< sample to skip during training
    uint32_t trainPasses = 1; ///< number of training passes over the training data
    bool startWithFullWindow = false; ///< if true skip the smaller windows at the beginning of training
    bool llsInitialization = false; ///> should the linear weights be initialized with the LLS solution?
    bool trainingShuffle = false; ///> should the training data be shuffled?
    uint32_t dictionarySize; ///< maximum number of (complex) basis vectors to be saved in the dictionary
    RealSample normConstraint; ///< norm constraint to use for dictionary sparsification (set to 0 to disable)
    RealSample eB; ///< hyperslab width
};

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "Parameters = " << par << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const apsm_parameters& par );

/**
 * @}
 */
