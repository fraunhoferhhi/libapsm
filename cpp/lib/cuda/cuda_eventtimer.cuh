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
 * @file cuda_eventtimer.cuh
 * @brief CUDA event base time measurement header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.06.24   0.01    initial version
 */

// include guard
#pragma once

// STD C
#include <assert.h>

// CUDA
#include <cuda_runtime.h>

// APSM
#include "cuda/cuda_errorhandling.cuh"

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

/**
 * @brief Kernel processing time measurement
 */
class CUDA_EventTimer
{

public:
    CUDA_EventTimer()
        : mStarted( false )
        , mStopped( false )
    {
        CUDA_CHECK( cudaEventCreate( &mStart ) );
        CUDA_CHECK( cudaEventCreate( &mStop ) );
    }

    ~CUDA_EventTimer()
    {
        CUDA_CHECK( cudaEventDestroy( mStart ) );
        CUDA_CHECK( cudaEventDestroy( mStop ) );
    }

    /**
     * @brief Start time measurement
     *
     * @param s
     */
    void start( cudaStream_t s = 0 )
    {
        CUDA_CHECK( cudaEventRecord( mStart, s ) );
        mStarted = true;
        mStopped = false;
    }

    /**
     * @brief Stop time measurement
     *
     * @param s
     */
    void stop( cudaStream_t s = 0 )
    {
        assert( mStarted );
        CUDA_CHECK( cudaEventRecord( mStop, s ) );
        mStarted = false;
        mStopped = true;
    }

    /**
     * @brief Calculate time difference
     *
     * @return time difference
     */
    float elapsed()
    {
        assert( mStopped );
        CUDA_CHECK( cudaEventSynchronize( mStop ) );
        float elapsed = 0;
        CUDA_CHECK( cudaEventElapsedTime( &elapsed, mStart, mStop ) );
        return elapsed;
    }

private:
    bool mStarted, mStopped;
    cudaEvent_t mStart, mStop;
};

/**
 * @}
 */
