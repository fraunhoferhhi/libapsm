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
 * @file util.cpp
 * @brief util functions
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.xx.xx   0.01    initial version
 */

// STD C
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <random>

// APSM - Command Line Interface (CLI)
#include "util.hpp"

float printSymbolErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation )
{
    const float modScale = modulation.getScale();
    uint32_t symbol_errors = 0;
    const float maxConstellationVal = ( modulation.getBitPerSymbol() == 1 ) ? 1 : ( ( 1 << ( modulation.getBitPerSymbol() / 2 ) ) - 1 ) * modScale;

    for ( uint32_t ldx = 0; ldx < estimatedSamples.size(); ldx++ )
    {
        ComplexSample y = estimatedSamples[ ldx ];

        // linewrap after 80 chars
        if ( ldx % 80 == 0 )
            std::cout << std::endl;

        if ( std::isnan( std::abs( y ) ) )
        {
            symbol_errors++;
            std::cout << "n";
        }
        else if ( std::abs( y.real() - trueSamples[ ldx ].real() ) > modScale
                  && !( y.real() > +maxConstellationVal
                        && trueSamples[ ldx ].real() > +maxConstellationVal - modScale )
                  && !( y.real() < -maxConstellationVal
                        && trueSamples[ ldx ].real() < -maxConstellationVal + modScale ) )
        {
            symbol_errors++;
            std::cout << "r";
        }
        else if ( std::abs( y.imag() - trueSamples[ ldx ].imag() ) > modScale
                  && !( y.imag() > +maxConstellationVal
                        && trueSamples[ ldx ].imag() < +maxConstellationVal - modScale )
                  && !( y.imag() < -maxConstellationVal
                        && trueSamples[ ldx ].imag() > -maxConstellationVal + modScale )
                  && modulation.getBitPerSymbol() > 1 )
        {
            symbol_errors++;
            std::cout << "i";
        }
        else
        {
            std::cout << ".";
        }
    }
    uint32_t total_symbols = estimatedSamples.size();
    float symbol_error_percentage = 100. * symbol_errors / total_symbols;
    std::cout << " symbol errors: " << symbol_errors << " of " << total_symbols << " - " << std::fixed << std::setprecision( 2 ) << symbol_error_percentage << " percent" << std::endl;

    return symbol_error_percentage;
}

float printBitErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation )
{
    std::vector<bool> estimatedBits = modulation.hardDecode( estimatedSamples );
    std::vector<bool> trueBits = modulation.hardDecode( trueSamples );

    uint32_t totalBits = estimatedBits.size();
    uint32_t bitErrors = 0;
    for ( uint32_t idx = 0; idx < totalBits; idx++ )
    {
        if ( idx % ( modulation.getBitPerSymbol() * 80 ) == 0 )
            std::cout << std::endl;
        if ( idx % modulation.getBitPerSymbol() == 0 )
            std::cout << " ";

        if ( estimatedBits[ idx ] != trueBits[ idx ] )
        {
            bitErrors++;
            std::cout << "x";
        }
        else
        {
            std::cout << ".";
        }
    }

    float bitErrorPercentage = 100. * bitErrors / totalBits;
    std::cout << " bit errors: " << bitErrors << " of " << totalBits << " - " << std::fixed << std::setprecision( 2 ) << bitErrorPercentage << " percent" << std::endl;

    return bitErrorPercentage;
}

float printEvm( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples )
{
    const float fullBarValue = 0.5F; // For Error Magnitudes above this level, a full bar is shown
    const uint32_t totalSymbols = estimatedSamples.size();
    float squaredSum = 0;

    for ( uint32_t idx = 0; idx < totalSymbols; idx++ )
    {
        // linewrap after 80 chars
        if ( idx % 80 == 0 )
            std::cout << std::endl;

        float error = std::abs( estimatedSamples[ idx ] - trueSamples[ idx ] );
        squaredSum += error * error;
        switch ( uint32_t( error / fullBarValue * 8 ) )
        {
        case 0:
            std::cout << " ";
            break;
        case 1:
            std::cout << "\u2581";
            break;
        case 2:
            std::cout << "\u2582";
            break;
        case 3:
            std::cout << "\u2583";
            break;
        case 4:
            std::cout << "\u2584";
            break;
        case 5:
            std::cout << "\u2585";
            break;
        case 6:
            std::cout << "\u2586";
            break;
        case 7:
            std::cout << "\u2587";
            break;
        default:
            std::cout << "\u2588";
            break;
        }
    }

    float evm = sqrt( squaredSum / totalSymbols );
    std::cout << " EVM: " << std::fixed << std::setprecision( 4 ) << evm << std::endl;
    return evm;
}

void printErrors( OutputMode outputMode, ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation )
{
    switch ( outputMode )
    {
    case BER:
        printBitErrors( estimatedSamples, trueSamples, modulation );
        break;
    case SER:
        printSymbolErrors( estimatedSamples, trueSamples, modulation );
        break;
    case EVM:
        printEvm( estimatedSamples, trueSamples );
        break;
    case Scripting:
        float bitErrors = printBitErrors( estimatedSamples, trueSamples, modulation );
        float symbolErrors = printSymbolErrors( estimatedSamples, trueSamples, modulation );
        float evm = printEvm( estimatedSamples, trueSamples );
        std::cout.clear(); // Temporary enable output
        std::cout << std::setprecision( -1 ) << bitErrors << "," << symbolErrors << "," << evm << std::endl;
        std::cout.setstate( std::ios_base::failbit ); // Disable all outputs for quiet mode
        break;
    }
}

AntennaPattern jsonGetAntennaPattern( const nlohmann::json& config )
{
    auto validKeys = { "otype", "pattern", "number" };
    for ( auto& [ key, value ] : config.items() )
    {
        if ( std::find( validKeys.begin(), validKeys.end(), key ) == validKeys.end() )
            throw std::invalid_argument( "Invalid key: " + key );
    }

    if ( config[ "otype" ] == "all" )
        return AntennaPattern().set();
    else if ( config[ "otype" ] == "pattern" )
        return AntennaPattern( config.value( "pattern", "11111111111111111111" ) );
    else if ( config[ "otype" ] == "equidistant" )
        return antennaPatternEquidistant( config.value( "number", 16 ) );
    else if ( config[ "otype" ] == "random" )
        return antennaPatternRandom( config.value( "number", 16 ) );
    else if ( config[ "otype" ] == "first" )
        return antennaPatternFirst( config.value( "number", 16 ) );
    else
        throw std::invalid_argument( "Invalid antenna scheme: " + config[ "otype" ].get<std::string>() );
}

AntennaPattern antennaPatternRandom( uint32_t numAntennas )
{
    std::vector<uint32_t> antennas( MAX_ANTENNAS );
    for ( uint32_t idx = 0; idx < MAX_ANTENNAS; idx++ )
        antennas[ idx ] = idx;

    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( antennas.begin(), antennas.end(), g );

    AntennaPattern antennaPattern;
    for ( uint32_t idx = 0; idx < numAntennas; idx++ )
        antennaPattern.set( antennas[ idx ] );

    return antennaPattern;
}

AntennaPattern antennaPatternFirst( uint32_t numAntennas )
{
    AntennaPattern antennaPattern;
    for ( uint32_t idx = 0; idx < numAntennas; idx++ )
        antennaPattern.set( idx );
    return antennaPattern;
}

AntennaPattern antennaPatternEquidistant( uint32_t numAntennas )
{
    AntennaPattern antennaPattern;
    for ( uint32_t idx = 0; idx < numAntennas; idx++ )
        antennaPattern.set( idx * MAX_ANTENNAS / numAntennas );
    return antennaPattern;
}

ComplexSampleMatrix selectAntennas( const ComplexSampleMatrix& input, const AntennaPattern& antennaPattern )
{
    const uint32_t numAntennas = std::min( input.size(), antennaPattern.size() );
    ComplexSampleMatrix output;
    output.reserve( antennaPattern.count() );
    for ( uint32_t idx = 0; idx < numAntennas; idx++ )
    {
        if ( antennaPattern.test( idx ) )
            output.push_back( input[ idx ] );
    }
    return output;
}

std::vector<bool> getRandomBits( uint32_t seed, uint32_t numBits )
{
    std::mt19937 gen( seed );
    std::bernoulli_distribution d( 0.5 );
    std::vector<bool> result;
    result.reserve( numBits );
    for ( uint32_t idx = 0; idx < numBits; idx++ )
        result.push_back( d( gen ) );
    return result;
}

ComplexSampleVector getRandomSymbols( uint32_t seed, const Modulation& modulation, uint32_t numSymbols )
{
    std::vector<bool> bits = getRandomBits( seed, numSymbols * modulation.getBitPerSymbol() );
    ComplexSampleVector symbols = modulation.encode( bits );
    return symbols;
}

void generateReferenceData( ComplexSampleMatrix& trainingSymbols, ComplexSampleMatrix& dataSymbols, uint32_t seed, const Modulation& modulation, uint32_t numUsers, uint32_t numTrainingSamples, uint32_t numDataSamples )
{

    if ( numUsers == 0 )
        numUsers = CONSTANTS::defaultNumUser;
    if ( numTrainingSamples == 0 )
        numTrainingSamples = CONSTANTS::defaultTrainingSamples;
    if ( numDataSamples == 0 )
        numDataSamples = CONSTANTS::defaultDataSamples;

    ComplexSampleVector symbols = getRandomSymbols( seed, modulation, ( numTrainingSamples + numDataSamples ) * numUsers );

    trainingSymbols.reserve( numUsers );
    dataSymbols.reserve( numUsers );
    for ( uint32_t u = 0; u < numUsers; u++ )
    {
        auto userIter = symbols.begin() + u * ( numTrainingSamples + numDataSamples );
        trainingSymbols.push_back( ComplexSampleVector( userIter, userIter + numTrainingSamples ) );
        dataSymbols.push_back( ComplexSampleVector( userIter + numTrainingSamples, userIter + numTrainingSamples + numDataSamples ) );
    }
}
