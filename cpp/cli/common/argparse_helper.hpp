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
 * @file argparse_helper.hpp
 * @brief helper for argument parser header
 *
 * @author Danie Schäufele    HHI,
 *
 * @date 2022.08.03   0.01    initial version
 */

// include guard
#pragma once

// ARGPARSE
#include <argparse/argparse.hpp>

// APSM - Command Line Interface (CLI)
#include "util.hpp"

class ArgparseHelper : public argparse::ArgumentParser
{
public:
    using argparse::ArgumentParser::ArgumentParser;

    void addParamRefData( bool modulationFromJson = false );
    std::tuple<ComplexSampleMatrix, ComplexSampleMatrix> loadRefData( bool modulationFromJson = false );

    void addParamTxData();
    ComplexSampleMatrix loadTxData();
    void writeTxData( const ComplexSampleMatrix& );

    void addParamRxData();
    ComplexSampleMatrix loadRxData();
    void writeRxData( const ComplexSampleMatrix& );

    void addParamSyncedData();
    void writeSyncedData( const ComplexSampleMatrix&, const ComplexSampleMatrix& );
    std::tuple<ComplexSampleMatrix, ComplexSampleMatrix> loadSyncedData();

    void addParamDetectedData();
    void writeDetectedData( const ComplexSampleVector& );

    void addParamTransmissionMode();
    bool getOfdmEnabled();

    void addParamOutputMode();
    OutputMode getOutputMode();

    void addParamConfig();
    nlohmann::json getConfig();

    void parse_args_check_exit( int argc, char* argv[] );
};
