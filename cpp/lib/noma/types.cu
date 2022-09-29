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
 * @file types.cu
 * @brief NOMA data types
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2021.12.03   0.01    initial version
 */

// STD C
#include <type_traits>

// APSM
#include "noma/types.cuh"

// check that RealSample is from type float
static_assert( std::is_same_v<RealSample, float>, "Only float data is supported" );

/**
 * @brief Prints a human readable representation of RealSampleVector to a stream.
 *
 * @param os Stream to print to
 * @param m RealSampleVector that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const RealSampleVector& m )
{

    os << "  [";

    for ( uint32_t idx = 0; idx < m.size(); idx++ )
    {
        os << m[ idx ];
        if ( idx < m.size() - 1 )
            os << ", ";
    }
    os << "]" << std::endl;

    return os;
}

/**
 * @brief Prints a human readable representation of RealSampleMatrix to a stream.
 *
 * @param os Stream to print to
 * @param m RealSampleMatrix that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const RealSampleMatrix& m )
{
    os << "[" << std::endl;
    for ( uint32_t idx = 0; idx < m.size(); idx++ )
    {
        os << "  [";
        for ( uint32_t jdx = 0; jdx < m[ idx ].size(); jdx++ )
        {
            os << m[ idx ][ jdx ];
            if ( jdx < m[ idx ].size() - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}

/**
 * @brief Prints a human readable representation of ComplexSampleVector to a stream.
 *
 * @param os Stream to print to
 * @param m ComplexSampleVector that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const ComplexSampleVector& m )
{

    os << "  [";

    for ( uint32_t idx = 0; idx < m.size(); idx++ )
    {
        os << m[ idx ];
        if ( idx < m.size() - 1 )
            os << ", ";
    }
    os << "]" << std::endl;

    return os;
}

/**
 * @brief Prints a human readable representation of ComplexSampleMatrix to a stream.
 *
 * @param os Stream to print to
 * @param m ComplexSampleMatrix that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const ComplexSampleMatrix& m )
{
    os << "[" << std::endl;
    for ( uint32_t idx = 0; idx < m.size(); idx++ )
    {
        os << "  [";
        for ( uint32_t jdx = 0; jdx < m[ idx ].size(); jdx++ )
        {
            os << m[ idx ][ jdx ];
            if ( jdx < m[ idx ].size() - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}
