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
 * @file cuda_matrix_device.cuh
 * @brief CUDA matrix multiplication header
 *
 * @author Daniel Schäufele, HHI,
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.10.29   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 * @date 2020.07.17   0.03    refactoring
 */

// include guard
#pragma once

// APSM
#include "cuda/cuda_types.cuh"

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

/**
 * @class CudaDeviceMatrix
 *
 * @brief Class that handles a matrix on device.
 *
 * @details
 * All data in this class is saved on the device (GPU).
 */
class CudaDeviceMatrix
{

public:
    /**
     * @brief A constructor for ... .
     */
    CudaDeviceMatrix( const uint32_t _height, const uint32_t _width, const RealSample* _elements )
        : height( _height )
        , width( _width )
        , elements( _elements )
    {
    }

    /**
     * @brief Return the height (number of rows)
     *
     * @return uint32_t (height/rows)
     */
    __device__ uint32_t getHeight() const
    {
        return height;
    }

    /**
     * @brief Get the width (number of columns)
     *
     * @return uint32_t (width/columns)
     */
    __device__ uint32_t getWidth() const
    {
        return width;
    }

    /**
     * @brief Return a reference to a single element on a device matrix.
     *
     * @param[in] row
     * @param[in] col
     * @return reference to element
     */
    __device__ RealSample&
    operator()( uint32_t row, uint32_t col = 0 )
    {
        assert( row < height );
        assert( col < width );
        return const_cast<RealSample&>( elements[ row * width + col ] );
    }

    /**
     * @brief Return a constant reference to a single element on a device matrix.
     *
     * @param[in] row
     * @param[in] col
     * @return const_reference to element
     */
    __device__ const RealSample&
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        assert( row < height );
        assert( col < width );
        return elements[ row * width + col ];
    }

protected:
    // protected variables
    const uint32_t height; ///< device matrix height
    const uint32_t width; ///< device matrix width
    const RealSample* const elements; ///< device matrix pointer to elements
};

/**
 * @class CudaDeviceDedupMatrix
 *
 * @brief Class for matrix duplication on device.
 *
 * @details Here is a basic example.
 *
 * referenced device matrix as input:
 * | Re(A)  Im(A)  Re(B)  Im(B) |
 * | Re(C)  Im(C)  Re(D)  Im(D) |
 *
 * output:
 * | Re(A)  Im(A)  Re(B)  Im(B) |
 * | Re(C)  Im(C)  Re(D)  Im(D) |
 * | Im(A) -Re(A)  Im(B) -Re(B) | <-- swap pair of Re and Im at this half of rows
 * | Im(C) -Re(C)  Im(D) -Re(D) | <-- and minus Re at this half of rows
 */
class CudaDeviceDedupMatrix : public CudaDeviceMatrix
{

public:
    // inherit constructors
    using CudaDeviceMatrix::CudaDeviceMatrix;

    /**
     * @brief Returns the corresbonding matrix duplication element in given row and column.
     *
     * @details Duplicate row in the given matrix.
     *          Also swap Real and Imaginary part in column and negate Real part in the second half of rows.
     *
     * @param[in] row
     * @param[in] col
     * @return RealSample
     */
    __device__ RealSample
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        int32_t swap = 1 - 2 * ( col % 2 );
        int32_t DedupSign = ( row < height ) ? 1 : swap; // minus at Re(.)
        // int32_t DedupSign = ( row < height ) ? 1 : -swap; // minus at Im(.)

        return DedupSign * CudaDeviceMatrix::operator()( row % height, ( row < height ) ? col : col + swap );
    }

    /**
     * @brief Member function whicht returns the height of the matrix duplication object.
     *
     * @return uint32_t (height)
     */
    __device__ uint32_t getHeight() const
    {
        return height * 2;
    }
};

/**
 * @class CudaDeviceRingBuffer
 *
 * @brief Class for ringbuffer on device.
 */
class CudaDeviceRingBuffer : public CudaDeviceMatrix
{

public:
    /**
     * @brief A constructor for a new CudaDeviceRingBuffer object with with defined capacity.
     */
    CudaDeviceRingBuffer( const uint32_t _height, const uint32_t _width, const RealSample* _elements, const uint32_t _offset, const uint32_t _usedHeight, const uint32_t _windowHeight )
        : CudaDeviceMatrix( _height, _width, _elements )
        , offset( _offset )
        , usedHeight( _usedHeight )
        , windowHeight( _windowHeight )
    {
    }

    /**
     * @brief Returns the height (number of rows).
     *
     * @return uint32_t (height)
     */
    __device__ uint32_t getUsedHeight() const
    {
        return usedHeight > windowHeight ? windowHeight : usedHeight;
    }

    /**
     * @brief Return a reference to a single element on a device ringbuffer.
     *
     * @param[in] row
     * @param[in] col
     * @return reference to element
     */
    __device__ RealSample&
    operator()( uint32_t row, uint32_t col = 0 )
    {
        assert( row < usedHeight );
        assert( col < width );
        return CudaDeviceMatrix::operator()( ( row + offset ) % height, col );
    }

    /**
     * @brief Return a constant reference to a single element on a device ringbuffer.
     *
     * @param[in] row
     * @param[in] col
     * @return const_reference to element
     */
    __device__ const RealSample&
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        assert( row < usedHeight );
        assert( col < width );
        return CudaDeviceMatrix::operator()( ( row + offset ) % height, col );
    }

protected:
    // protected variables
    const uint32_t offset; ///< Offset
    const uint32_t usedHeight; ///< Used height
    const uint32_t windowHeight; ///< Window height
};

/**
 * @class CudaDeviceDedupRingBuffer
 *
 * @brief Class for ringbuffer duplication on device.
 */
class CudaDeviceDedupRingBuffer : public CudaDeviceRingBuffer
{
public:
    using CudaDeviceRingBuffer::CudaDeviceRingBuffer; // inherit constructors

    /**
     * @brief Returns the corresbonding ringbuffer duplication element in given row and column.
     *
     * @details Duplicate column in the given ringbuffer.
     *          Also swap Real and Imaginary part in rows and negate Real part in the second half of columns.
     *
     * @param[in] row
     * @param[in] col
     * @return RealSample
     */
    __device__ RealSample
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        int32_t swap = 1 - 2 * ( row % 2 );
        int32_t DedupSign = ( col < width ) ? 1 : swap; // minus at Re(.)
        // int32_t DedupSign = ( col < width ) ? 1 : -swap; // minus at Im(.)

        return DedupSign * CudaDeviceRingBuffer::operator()( ( col < width ) ? row : row + swap, col % width );
    }

    /**
     * @brief Member function that returns the width of the ringbuffer duplication object.
     *
     * @return uint32_t (width)
     */
    __device__ uint32_t getWidth() const
    {
        return width * 2;
    }
};

/**
 * @class DeviceTrainingState
 *
 * @brief Class for Training state on device.
 */
class DeviceTrainingState
{

public:
    /**
     * @brief A constructor for a new DeviceTrainingState object with given ringbuffer content for basis and coefficients.
     */
    DeviceTrainingState( const CudaDeviceDedupRingBuffer& _basis, const CudaDeviceRingBuffer& _gaussianCoeffs, const CudaDeviceMatrix& _linearCoeffs )
        : basis( _basis )
        , gaussianCoeffs( _gaussianCoeffs )
        , linearCoeffs( _linearCoeffs )
    {
    }

    // public variables
    CudaDeviceDedupRingBuffer basis; ///< Basis ringbuffer
    CudaDeviceRingBuffer gaussianCoeffs; ///< Gaussian coefficient ringbuffer
    CudaDeviceMatrix linearCoeffs; ///< Linear coefficient ringbuffer */
};

/**
 * @class DeviceStatisticState
 *
 * @brief Class for Statistics state on device.
 */
class DeviceStatisticState
{

public:
    /**
     * @brief A constructor for a new empty DeviceStatisticState object.
     */
    DeviceStatisticState() { }

    /**
     * @brief A constructor for a new DeviceStatisticState object with given bitErrors, symbolErrors and evm content.
     */
    DeviceStatisticState( const float& _bitErrors, const float& _symbolErrors, const float& _evm )
        : bitErrors( _bitErrors )
        , symbolErrors( _symbolErrors )
        , evm( _evm )
    {
    }

    // public variables
    RealSample bitErrors; ///< BIT error
    RealSample symbolErrors; ///< OFDM Symbol error
    RealSample evm; ///< Error Vector Magnitude
};

/**
 * @}
 */
