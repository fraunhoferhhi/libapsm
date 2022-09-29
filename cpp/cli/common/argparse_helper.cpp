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
 * @file argparse_helper.cpp
 * @brief helper for argument parser
 *
 * @author Danie Schäufele    HHI,
 *
 * @date 2022.08.03   0.01    initial version
 */

// STD C
#include <fstream>

// APSM - Command Line Interface (CLI)
#include "argparse_helper.hpp"
#include "binary_load.hpp"

void ArgparseHelper::addParamRefData( bool modulationFromJson )
{
    add_argument( "-r", "--reference-data" )
        .help( "reference data file" );

    add_argument( "-rs", "--random-seed" )
        .help( "random seed for generation of data (this overrides the --reference-data argument)" )
        .action( []( const std::string& value ) { return (uint32_t)std::stoul( value ); } );

    if ( !modulationFromJson )
    {
        add_argument( "-m", "--modulation" )
            .default_value( std::string( "QAM16" ) )
            .required()
            .help( "modulation scheme" );
    }
}

std::tuple<ComplexSampleMatrix, ComplexSampleMatrix> ArgparseHelper::loadRefData( bool modulationFromJson )
{
    if ( !present<uint32_t>( "--random-seed" ) && !present<std::string>( "--reference-data" ) )
        throw std::invalid_argument( "Neither --random-seed/-rs nor --reference-data/-r was specified" );

    Modulation modulation = Modulation::off;
    if ( modulationFromJson )
    {
        auto config = getConfig();
        modulation = Modulation::fromString( config.value( "modulation", "QAM16" ) );
    }
    else
    {
        modulation = Modulation::fromString( get<std::string>( "--modulation" ) );
    }
    ComplexSampleMatrix refSigTraining, refSigData;
    if ( present<uint32_t>( "--random-seed" ) )
    {
        uint32_t randomSeed = get<uint32_t>( "--random-seed" );
        generateReferenceData( refSigTraining, refSigData, randomSeed, modulation ); //, numUsers, numTrainingSamples, numDataSamples );
    }
    else
    {
        std::string refDataFile = get<std::string>( "--reference-data" );
        loadData( refDataFile, "mod_training", refSigTraining, modulation.getScale() ); // , numUsers, numTrainingSamples
        loadData( refDataFile, "mod_data", refSigData, modulation.getScale() ); // , numUsers, numDataSamples
    }
    return { refSigTraining, refSigData };
}

void ArgparseHelper::addParamTxData()
{
    add_argument( "-t", "--transmitter-data" )
        .required()
        .help( "transmitter data file" );
}

ComplexSampleMatrix ArgparseHelper::loadTxData()
{
    ComplexSampleMatrix data;
    loadData( get<std::string>( "--transmitter-data" ), "txsignal", data );
    return data;
}

void ArgparseHelper::writeTxData( const ComplexSampleMatrix& txSignal )
{
    BinaryFileWriter fileWriter( get<std::string>( "--transmitter-data" ) );
    fileWriter.addData( "txsignal", txSignal );
    fileWriter.write();
}

void ArgparseHelper::addParamRxData()
{
    add_argument( "-rx", "--received-data" )
        .required()
        .help( "received data file" );
}

ComplexSampleMatrix ArgparseHelper::loadRxData()
{
    ComplexSampleMatrix data;
    loadData( get<std::string>( "--received-data" ), "rxSig", data );
    return data;
}

void ArgparseHelper::writeRxData( const ComplexSampleMatrix& data )
{
    BinaryFileWriter fileWriter( get<std::string>( "--received-data" ) );
    fileWriter.addData( "rxSig", data );
    fileWriter.write();
}

void ArgparseHelper::addParamSyncedData()
{
    add_argument( "-s", "--synchronized-data" )
        .required()
        .help( "synchronized data file containing received training and data signals" );
}

void ArgparseHelper::writeSyncedData( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleMatrix& rxSigData )
{
    BinaryFileWriter fileWriter( get<std::string>( "--synchronized-data" ) );
    fileWriter.addData( "rxSigTraining", rxSigTraining );
    fileWriter.addData( "rxSigData", rxSigData );
    fileWriter.write();
}

std::tuple<ComplexSampleMatrix, ComplexSampleMatrix> ArgparseHelper::loadSyncedData()
{
    ComplexSampleMatrix rxSigTraining, rxSigData;
    std::string syncedDataFile = get<std::string>( "--synchronized-data" );
    loadData( syncedDataFile, "rxSigTraining", rxSigTraining );
    loadData( syncedDataFile, "rxSigData", rxSigData );
    return { rxSigTraining, rxSigData };
}
void ArgparseHelper::addParamDetectedData()
{
    add_argument( "-d", "--detected-data" )
        .help( "detected (estimated) data file" );
}

void ArgparseHelper::writeDetectedData( const ComplexSampleVector& data )
{
    if ( present<std::string>( "--detected-data" ) )
    {
        BinaryFileWriter fileWriter( get<std::string>( "--detected-data" ) );
        fileWriter.addData( "estSigData", data );
        fileWriter.write();
    }
}

void ArgparseHelper::addParamTransmissionMode()
{
    add_argument( "-tm", "--transmission-mode" )
        .default_value( std::string( "TIME" ) )
        .required()
        .help( "transmission mode" );
}

bool ArgparseHelper::getOfdmEnabled()
{
    std::string mode = get<std::string>( "--transmission-mode" );
    std::transform( mode.begin(), mode.end(), mode.begin(), ::toupper );
    if ( mode == "TIME" )
        return false;
    else if ( mode == "OFDM" )
        return true;
    else
        throw std::invalid_argument( "Unknown transmission mode: " + get<std::string>( "--transmission-mode" ) );
}

void ArgparseHelper::addParamOutputMode()
{
    add_argument( "-o", "--output-mode" )
        .default_value( std::string( "SER" ) )
        .required()
        .help( "print mode for error metric (BER = bit errors, SER = symbol errors, EVM = Error Vector Magnitude, scripting = disable all output except \"ber,ser,evm\")" );
}

OutputMode ArgparseHelper::getOutputMode()
{
    std::string outputModeStr = get<std::string>( "--output-mode" );
    std::transform( outputModeStr.begin(), outputModeStr.end(), outputModeStr.begin(), ::toupper );
    if ( outputModeStr == "BER" )
        return BER;
    else if ( outputModeStr == "SER" )
        return SER;
    else if ( outputModeStr == "EVM" )
        return EVM;
    else if ( outputModeStr == "SCRIPTING" )
    {
        std::cout.setstate( std::ios_base::failbit ); // Disable all outputs for quiet mode
        return Scripting;
    }
    else
        throw std::invalid_argument( "Invalid print mode: " + get<std::string>( "--output-mode" ) );
}

void ArgparseHelper::addParamConfig()
{
    add_argument( "-cf", "--config-file" )
        .help( "filename for JSON file containing the algorithm configuration" );

    add_argument( "-cs", "--config-string" )
        .default_value( std::string( "{ \"algorithm\": { \"otype\": \"APSM\" } }" ) )
        .required()
        .help( "JSON string containing the algorithm configuration" );
}

nlohmann::json ArgparseHelper::getConfig()
{
    if ( present<std::string>( "--config-file" ) )
    {
        std::ifstream f( get<std::string>( "--config-file" ) );
        return nlohmann::json::parse( f );
    }
    else
    {
        return nlohmann::json::parse( get<std::string>( "--config-string" ) );
    }
}

void ArgparseHelper::parse_args_check_exit( int argc, char* argv[] )
{
    try
    {
        parse_args( argc, argv );
    }
    catch ( const std::runtime_error& err )
    {
        std::cerr << err.what() << std::endl;
        std::cerr << *this;
        std::exit( EXIT_FAILURE );
    }
}
