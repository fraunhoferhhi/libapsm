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
 * @file    noma_detect.cpp
 * @brief   APSM command line interface (cli) tool
 * @details ... tbd
 *
 * @author  Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date    2020.09.29   0.01    initial version
 * @date    2020.10.14   0.02    remove boost dependencies
 */

// STD C
#include <thread>
#include <vector>

// APSM
#include <noma/noma_detector_factory.cuh>

// APSM - Command Line Interface (CLI)
#include "common/argparse_helper.hpp"
#include "common/util.hpp"

int main( int argc, char* argv[] )
{
    ArgparseHelper arpa( "NOMA_detect", "0.0.3" );
    arpa.add_description( "This tool runs the training and detection algorithms. The algorithm configuration is passed in JSON format either as string or in a file." );
    arpa.addParamSyncedData();
    arpa.addParamRefData( true );
    arpa.addParamDetectedData();
    arpa.addParamConfig();
    arpa.addParamOutputMode();

    arpa.add_argument( "-th", "--train-history-file" )
        .default_value( std::string() )
        .help( "filename for training history CSV-file (test set will be used as validation set)" );

    arpa.add_argument( "-p", "--parallel-processing" )
        .help( "process all users in parallel" )
        .default_value( false )
        .implicit_value( true );

    arpa.parse_args_check_exit( argc, argv );

    auto [ rxSigTraining, rxSigData ] = arpa.loadSyncedData();
    auto [ txSigTraining, txSigData ] = arpa.loadRefData( true );
    auto config = arpa.getConfig();
    OutputMode outputMode = arpa.getOutputMode();
    std::string trainHistoryFile = arpa.get<std::string>( "--train-history-file" );

    uint32_t user = config.value( "user", 0 );
    Modulation modulation = Modulation::fromString( config.value( "modulation", "QAM16" ) );
    nlohmann::json defaultAntennaConfig = { { "otype", "all" } };
    AntennaPattern antennaPattern = jsonGetAntennaPattern( config.value( "antennas", defaultAntennaConfig ) );

    rxSigTraining = selectAntennas( rxSigTraining, antennaPattern );
    rxSigData = selectAntennas( rxSigData, antennaPattern );

    if ( arpa.get<bool>( "--parallel-processing" ) )
    {
        std::vector<std::thread> threads;
        std::vector<ComplexSampleVector> estSigDatas( CONSTANTS::defaultNumUser );
        std::vector<std::unique_ptr<NomaDetector>> detectors;
        detectors.reserve( CONSTANTS::defaultNumUser );

        for ( uint32_t idx = 0; idx < CONSTANTS::defaultNumUser; idx++ )
        {
            detectors.push_back( NomaDetectorFactory::build( config[ "algorithm" ], antennaPattern.count() ) );
            threads.push_back( std::thread( [ & ]( uint32_t idx ) {
            detectors[ idx ]->train( rxSigTraining, txSigTraining[ idx ] );
            detectors[ idx ]->detect( rxSigData, estSigDatas[ idx ] ); },
                                            idx ) );
        }

        for ( uint32_t idx = 0; idx < CONSTANTS::defaultNumUser; idx++ )
        {
            threads[ idx ].join();
            printErrors( outputMode, estSigDatas[ idx ], txSigData[ idx ], modulation );
        }
    }
    else
    {
        NomaValSet valSet = std::nullopt;
        NomaTrainHistory stats( modulation.getBitPerSymbol(), modulation.getScale() ); // define outside here, because otherwise it will be destructed at end of if block
        if ( !trainHistoryFile.empty() )
        {
            valSet.emplace( std::make_tuple( std::ref( stats ), std::cref( rxSigData ), std::cref( txSigData[ user ] ) ) );
        }

        // process data
        ComplexSampleVector estSigData;

        // call APSM wrapper
        std::unique_ptr<NomaDetector> d = NomaDetectorFactory::build( config[ "algorithm" ], antennaPattern.count() );
        float trainTime = d->train( rxSigTraining, txSigTraining[ user ], valSet );
        float detectTime = d->detect( rxSigData, estSigData );
        std::cout << "Train time = " << trainTime << " ms, detect time = " << detectTime << " ms" << std::endl;
        printErrors( outputMode, estSigData, txSigData[ user ], modulation );

        arpa.writeDetectedData( estSigData );

        if ( valSet.has_value() )
            std::get<0>( *valSet ).toCsv( trainHistoryFile );
    }

    return EXIT_SUCCESS;
}
