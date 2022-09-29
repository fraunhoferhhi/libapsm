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
 * @file modulation.hpp
 * @brief Modulation header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.11.14   0.01    initial version
 * @date 2020.06.30   0.02    switchable modulation schemes
 */

// include guard
#pragma once

// STD C
#include <algorithm>
#include <exception>

// APSM
#include <noma/types.cuh>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

class Modulation
{
#define SQRT( x ) ( sqrt( x ) )
#define INVSQRT( x ) ( 1.0 / sqrt( x ) )
#define divideBy3( x ) ( uint16_t )( ( (uint32_t)0xAAABUL * x ) >> 17 )

public:
    enum Type
    {
        off = 0, ///< OFF
        bpsk = 1, ///< BPSK, 1 bit per symbol
        qpsk = 2, ///< QPSK, 2 bits per symbol
        qam16 = 4, ///< QAM16, 4 bits per symbol
        qam64 = 6, ///< QAM64, 6 bits per symbol
        qam256 = 8, ///< QAM256, 8 bits per symbol
        qam1024 = 10, ///< QAM1024, 10 bits per symbol
        qam4096 = 12, ///< QAM4096, 12 bits per symbol
        qam16384 = 14, ///< QAM16384, 14 bits per symbol
    };

    Modulation( Type type )
        : type( type )
    {
        switch ( type )
        {
        case off:
            name = "OFF";
            break;
        case bpsk:
            name = "BPSK";
            break;
        case qpsk:
            name = "QPSK";
            break;
        case qam16:
            name = "QAM16";
            break;
        case qam64:
            name = "QAM64";
            break;
        case qam256:
            name = "QAM256";
            break;
        case qam1024:
            name = "QAM1024";
            break;
        case qam4096:
            name = "QAM4096";
            break;
        case qam16384:
            name = "QAM16384";
            break;
        default:
            throw std::invalid_argument( "Unsupported modulation" );
        }
    }

    static Modulation fromString( std::string name )
    {
        std::transform( name.begin(), name.end(), name.begin(), ::toupper );

        if ( name == "OFF" )
            return off;
        if ( name == "BPSK" )
            return bpsk;
        if ( name == "QPSK" )
            return qpsk;
        if ( name == "QAM16" )
            return qam16;
        if ( name == "QAM64" )
            return qam64;
        if ( name == "QAM256" )
            return qam256;
        if ( name == "QAM1024" )
            return qam1024;
        if ( name == "QAM4096" )
            return qam4096;
        if ( name == "QAM16384" )
            return qam16384;
        throw std::invalid_argument( "Unsupported modulation: " + name );
    }

    static Modulation fromBitPerSymbol( uint32_t bps )
    {
        return static_cast<Type>( bps );
    }

    // returns scaling factor which converts from FLOAT to INT
    RealSample getScale() const
    {

        switch ( type )
        {
        case off:
            return 0.;
        case bpsk:
        case qpsk:
            return INVSQRT( getBitPerSymbol() );
        case qam16:
        case qam64:
        case qam256:
        case qam1024:
        case qam4096:
        case qam16384:
            return INVSQRT( scalerValueQAM( getBitPerSymbol() ) );
        default:
            throw std::invalid_argument( "Unsupported modulation" );
        }
    }

    // returns scaling factor which converts from INT to FLOAT
    RealSample getRevScale() const
    {

        switch ( type )
        {
        case off:
            return 0.;
        case bpsk:
        case qpsk:
            return SQRT( getBitPerSymbol() );
        case qam16:
        case qam64:
        case qam256:
        case qam1024:
        case qam4096:
        case qam16384:
            return SQRT( scalerValueQAM( getBitPerSymbol() ) );
        default:
            throw std::invalid_argument( "Unsupported modulation" );
        }
    }

    std::string getName() const
    {
        return name;
    }

    uint32_t getBitPerSymbol() const
    {
        return static_cast<uint32_t>( type );
    }

    std::vector<bool> hardDecode( const ComplexSampleVector& input ) const;
    ComplexSampleVector encode( const std::vector<bool>& input ) const;

private:
    // Scaling expression for QAM Modulation
    constexpr uint32_t scalerValueQAM( uint32_t bps ) const
    {
        // NOTE: (mm) this only works with uint if variable "bps" is even and >= 2
        // ( 2 * exp2( bps ) - 2 ) / 3;
        return divideBy3( ( 2 << bps ) - 2 );
    }

protected:
    Type type;
    RealSample scale;
    std::string name;
};

/**
 * @}
 */
