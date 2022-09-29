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
 * @file apsm_versions.cuh
 * @brief APSM version control
 *
 * @author
 *
 * @date
 */

// include guard
#pragma once

// STD C
#include <string>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

enum apsm_function_id
{
    APSM_UNKNOWN_ID = 0,
    APSM_DETECT_ID = 1
};

constexpr uint32_t version_id( const enum apsm_function_id function, const int32_t version )
{
    return ( static_cast<int32_t>( function ) << 16 ) + version;
}

enum apsm_versions
{
    UNKNOWN = 0,
    // apsm_detect versions
    APSM_DETECT_ORIGINAL = version_id( APSM_DETECT_ID, 1 ),
    APSM_DETECT_OLDFAST = version_id( APSM_DETECT_ID, 2 ),
    APSM_DETECT_SHMEM = version_id( APSM_DETECT_ID, 6 ),
    APSM_DETECT_BALANCED = version_id( APSM_DETECT_ID, 7 )
};

inline const std::string apsm_get_version_string( const int32_t version_id )
{
    switch ( version_id )
    {
    case APSM_DETECT_ORIGINAL:
        return "APSM Detect original";
    case APSM_DETECT_OLDFAST:
        return "APSM Detect old fast code";
    case APSM_DETECT_SHMEM:
        return "APSM Detect Store vectors in shared memory";
    case APSM_DETECT_BALANCED:
        return "APSM Detect Balance memory and computation";
    default:
        return "Unknown";
    }
}

// select version to use
#ifndef APSM_DETECT_VERSION
#define APSM_DETECT_VERSION apsm_versions::APSM_DETECT_BALANCED
#endif

/**
 * @}
 */
