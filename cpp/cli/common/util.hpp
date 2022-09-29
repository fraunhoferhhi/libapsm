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
 * @file util.hpp
 * @brief util functions header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.xx.xx   0.01    initial version
 */

// include guard
#pragma once

// STD C
#include <bitset>

// JSON
#include <nlohmann/json.hpp>

// APSM
#include <noma/types.cuh>

// APSM - Command Line Interface (CLI)
#include "modulation.hpp"

class CONSTANTS
{
public:
    // FFT config
    static inline const uint32_t N_LCP = 160U; //< OFDM long cyclic prefix length
    static inline const uint32_t N_SCP = 144U; //< OFDM short cyclic prefix length
    static inline const uint32_t N_FFT = 2048U; //< OFDM symbol length (FFT size)

    // Derived from FFT config
    static inline const uint32_t N_LSYMBOL = N_FFT + N_LCP; //< Length of long OFDM symbol in samples
    static inline const uint32_t N_SSYMBOL = N_FFT + N_SCP; //< Length of short OFDM symbol in samples
    static inline const uint32_t N_SLOT = 1U * N_LSYMBOL + 6U * N_SSYMBOL; //< Length of an OFDM slot in samples
    static inline const uint32_t NUM_SAMPLES = 5U * N_SLOT; //< Length of our OFDM radioframe in samples
    static inline const uint32_t N_TRAIN = 5U * N_SSYMBOL; //< Length of Training sequence in samples
    static inline const uint32_t N_DECODE = 4U * N_SLOT; //< Length of Data sequence in samples

    // FIR filter (for time domain mode)
    static inline const uint32_t samples_per_symbol = 16U; //< FIR filter samples per symbol (2^4 oversampling)

    // Dataset generation
    static inline const uint32_t APSM_MAX_USER_ANT = 1U; //< currently each user have only one antenna
    static inline const uint32_t APSM_NOMA_LEN = 144U; //< Number of used subcarriers in OFDM mode <- Change number of used OFDM subcarriers here.

    // NUMBER of default OFDM symbols
    static inline const uint32_t TRAIN_SYMBOLS = 5U; //< Default number of OFDM symbols for training sequence
    static inline const uint32_t DECODE_SYMBOLS = 27U; //< Default number of OFDM symbols for data sequence

    // Define default number of test samples for training and data
    // max TIME mode
    static inline const uint32_t defaultTrainingSamples = N_TRAIN / samples_per_symbol; //< ( N_TRAIN / samples_per_symbol ) -> 5 OFDM symbols Default number of modulated training symbols  with 2048/144 FFT/CP size
    static inline const uint32_t defaultDataSamples = N_DECODE / samples_per_symbol; //< Default number of modulated data symbols ( N_DECODE / samples_per_symbol ) -> 4 x 7 OFDM symbols with 4x 2048/160 and all other 2048/144
    // max OFDM mode
    // static inline const uint32_t defaultTrainingSamples = APSM_NOMA_LEN * TRAIN_SYMBOLS; //< Default number of modulated training symbols  ( N_TRAIN / samples_per_symbol ) -> 5 OFDM symbols with 2048/144 FFT/CP size
    // static inline const uint32_t defaultDataSamples = APSM_NOMA_LEN * DECODE_SYMBOLS; //< Default number of modulated data symbols ( N_DECODE / samples_per_symbol ) -> 4 x 7 OFDM symbols with 4x 2048/160 and all other 2048/144

    // NUMBER of antennas on Tx and Rx side
    static inline const uint32_t defaultNumUser = 6U;
    static inline const uint32_t defaultNumRx = 16U;
};

enum OutputMode
{
    BER,
    SER,
    EVM,
    Scripting
};

const uint32_t MAX_ANTENNAS = 16;
typedef std::bitset<MAX_ANTENNAS> AntennaPattern;

// define GIT commit hash
#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "0000000" // 0000000 means uninitialized
#endif

float printSymbolErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation );
float printBitErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation );
float printEvm( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples );
void printErrors( OutputMode outputMode, ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const Modulation& modulation );

AntennaPattern jsonGetAntennaPattern( const nlohmann::json& config );
AntennaPattern antennaPatternRandom( uint32_t numAntennas );
AntennaPattern antennaPatternEquidistant( uint32_t numAntennas );
AntennaPattern antennaPatternFirst( uint32_t numAntennas );
ComplexSampleMatrix selectAntennas( const ComplexSampleMatrix& input, const AntennaPattern& antennaPattern );

std::vector<bool> getRandomBits( uint32_t seed, uint32_t numBits );
ComplexSampleVector getRandomSymbols( uint32_t seed, const Modulation& modulation, uint32_t numSymbols );
void generateReferenceData( ComplexSampleMatrix& trainingSymbols, ComplexSampleMatrix& dataSymbols, uint32_t seed, const Modulation& modulation, uint32_t numUsers = 0, uint32_t numTrainingSamples = 0, uint32_t numDataSamples = 0 );
