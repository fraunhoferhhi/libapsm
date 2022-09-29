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
 * @file types.cuh
 * @brief NOMA data types header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.01.06   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 * @date 2021.12.03   0.03    add stream operator
 */

// include guard
#pragma once

// STD C
#include <complex>
#include <iostream>
#include <vector>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

/** RealSample in single precision floating point calculation */
typedef float RealSample;

/** Vector for real samples or complex to real mapped samples */
typedef std::vector<RealSample> RealSampleVector;
/** Matrix for real samples or complex to real mapped samples */
typedef std::vector<RealSampleVector> RealSampleMatrix;

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "RealSampleVector = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const RealSampleVector& m );

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "RealSampleMatrix = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const RealSampleMatrix& m );

/** Complex sample */
typedef std::complex<RealSample> ComplexSample;

/** Vector for complex samples */
typedef std::vector<ComplexSample> ComplexSampleVector;
/** Matrix for complex samples */
typedef std::vector<ComplexSampleVector> ComplexSampleMatrix;

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "ComplexSampleVector = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const ComplexSampleVector& m );

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "ComplexSampleMatrix = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const ComplexSampleMatrix& m );

/**
 * @}
 */
