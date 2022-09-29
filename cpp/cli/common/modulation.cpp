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
 * @file modulation.cpp
 * @brief Modulation
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.11.14   0.01    initial version
 * @date 2020.06.30   0.02    switchable modulation schemes
 */

// STD C
#include <cassert>

// APSM - Command Line Interface (CLI)
#include "modulation.hpp"

/**
 * Demaps single dimension of QAM constellation to grey-decoded bits.
 * Symbols are placed at a distance of 2, starting at -num_symbols + 1 and ending at num_symbols - 1.
 * ATTENTION: Constellation is different from the one defined in the LTE standard.
 *
 * @param  x           input symbol
 * @param  num_symbols number of symbols
 * @return             decoded bits
 */
uint32_t demap_1d( const RealSample x, const uint32_t num_symbols )
{
    // convert ... -7 -5 -3 -1 1 3 5 7 ... to 0 1 2 3 4 5 6 7 ...
    int32_t xi = (int32_t)round( ( x + num_symbols - 1 ) / 2 );

    // limit to the valid range
    xi = std::min( (int32_t)num_symbols - 1, std::max( 0, xi ) );

    const uint32_t xi_gc = xi ^ ( xi >> 1 );
    return xi_gc;
}

uint32_t demap( const ComplexSample x, const uint32_t bps )
{
    if ( bps == 1 ) // special treatment for BPSK
        return demap_1d( x.real(), 2 );

    uint32_t num_symbols_1d = 1 << ( bps / 2 );
    uint32_t real_bits = demap_1d( x.real(), num_symbols_1d );
    uint32_t imag_bits = demap_1d( x.imag(), num_symbols_1d );
    return ( real_bits << ( bps / 2 ) ) | imag_bits;
}

std::vector<bool> Modulation::hardDecode( const ComplexSampleVector& input ) const
{
    std::vector<bool> output;
    output.reserve( input.size() * getBitPerSymbol() );

    RealSample revScale = getRevScale();

    for ( auto& sym : input )
    {
        uint32_t bits = demap( sym * revScale, getBitPerSymbol() );
        for ( uint32_t idx = 0; idx < getBitPerSymbol(); idx++ )
            output.push_back( bits & ( 1 << ( getBitPerSymbol() - idx - 1 ) ) );
    }
    return output;
}

RealSample map_1d( uint32_t bits, uint32_t num_symbols )
{
    uint32_t gray_bits = bits;
    while ( bits )
    {
        bits >>= 1;
        gray_bits ^= bits;
    }
    return 2 * (int32_t)gray_bits - (int32_t)num_symbols + 1;
}

ComplexSample map( uint32_t bits, const uint32_t bps )
{
    if ( bps == 1 ) // special treatment for BPSK
    {
        RealSample real = map_1d( bits, 2 );
        return ComplexSample( real, 0 );
    }
    else
    {
        uint32_t num_symbols_1d = 1 << bps / 2;
        RealSample real = map_1d( bits >> ( bps / 2 ), num_symbols_1d );
        RealSample imag = map_1d( bits & ( ( 1 << bps / 2 ) - 1 ), num_symbols_1d );
        return ComplexSample( real, imag );
    }
}

ComplexSampleVector Modulation::encode( const std::vector<bool>& input ) const
{
    assert( input.size() % getBitPerSymbol() == 0 );
    ComplexSampleVector output;
    output.reserve( input.size() / getBitPerSymbol() );

    uint32_t bits = 0;
    for ( uint32_t idx = 0; idx < input.size(); idx++ )
    {
        bits = bits << 1 | input[ idx ];
        if ( idx % getBitPerSymbol() == getBitPerSymbol() - 1 )
        {
            output.push_back( map( bits, getBitPerSymbol() ) * getScale() );
            bits = 0;
        }
    }
    return output;
}
