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
 * @file cuda_matrix_host.cuh
 * @brief APSM matrix multiplication header
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

// STD C
#include <iostream>

// APSM
#include "cuda/cuda_matrix_device.cuh"
#include "cuda/cuda_types.cuh"

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

#define IDX2E( row, col, rows ) ( ( ( col ) * ( rows ) ) + ( row ) )

/**
 * @class CudaHostMatrix
 *
 * @brief Class that handles a matrix on host.
 *
 * @details
 * Elements can be accessed by using the ()-operator, like this: `mat(i, j) = x;`.
 * The []-operator unfortunately only supports a single argument in C++.
 *
 * All data in this class is saved on the host (CPU).
 */
class CudaHostMatrix
{

public:
    /**
     * @brief A constructor for a new empty HostTrainingState object.
     */
    CudaHostMatrix() { }

    /**
     * @brief A constructor for ... .
     */
    CudaHostMatrix( uint32_t _height, uint32_t _width = 1, RealSample fill_value = 0 );

    /**
     * @brief A constructor for ... .
     */
    CudaHostMatrix( cudaStream_t stream, const ComplexSampleMatrix& data );

    /**
     * @brief A constructor for ... .
     */
    CudaHostMatrix( cudaStream_t stream, const ComplexSampleVector& data );

    /**
     * @brief A constructor for ... .
     */
    CudaHostMatrix( const RealSampleMatrix& data );

    /**
     * @brief Returns the height (number of rows).
     *
     * @return uint32_t (height/rows)
     */
    uint32_t getHeight() const
    {
        return height;
    }

    /**
     * @brief Returns the width (number of columns).
     *
     * @return uint32_t (width/columns)
     */
    uint32_t getWidth() const
    {
        return width;
    }

    uint32_t getNumElements() const
    {
        return getHeight() * getWidth();
    }

    /**
     * @brief ...
     */
    ThrustComplexSampleDeviceVector toComplexVector( cudaStream_t stream );

    bool operator==( const CudaHostMatrix& other ) const;

    /**
     * @brief Return a reference to a single element on a host matrix.
     *
     * @param[in] row
     * @param[in] col
     * @return reference to element
     */
    ThrustRealSampleDeviceVector::reference operator()( uint32_t row, uint32_t col = 0 )
    {
        assert( row < height );
        assert( col < width );
        return elements[ IDX2E( col, row, width ) ];
    }

    /**
     * @brief Return a constant reference to a single element on a host matrix.
     *
     * @param[in] row
     * @param[in] col
     * @return const_reference to element
     */
    ThrustRealSampleDeviceVector::const_reference operator()( uint32_t row, uint32_t col = 0 ) const
    {
        assert( row < height );
        assert( col < width );
        return elements[ IDX2E( col, row, width ) ];
    }

    /**
     * @brief Member function toDevice().
     *
     * @return CudaDeviceMatrix
     */
    CudaDeviceMatrix toDevice()
    {
        return CudaDeviceMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }

    /**
     * @brief Member function toDevice() with no change of objects check.
     *
     * @return const CudaDeviceMatrix
     */
    const CudaDeviceMatrix toDevice() const
    {
        return CudaDeviceMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }

    RealSample* getRawDataPointer()
    {
        return thrust::raw_pointer_cast( elements.data() );
    }

    const RealSample* getRawDataPointer() const
    {
        return thrust::raw_pointer_cast( elements.data() );
    }

    void copyToMemory( RealSample* dst, cudaStream_t stream ) const;

    RealSample* allocateAndCopyToMemory( cudaStream_t stream ) const;

    // allow access to private or protected information
    friend class CudaHostRingBuffer;
    friend std::ostream& operator<<( std::ostream& os, const CudaHostMatrix& m );

protected:
    // protected variables
    uint32_t height = 0; ///< host matrix height
    uint32_t width = 0; ///< host matrix width
    ThrustRealSampleDeviceVector elements; ///< host matrix elements
};

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "CudaHostMatrix = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const CudaHostMatrix& m );

/**
 * @class CudaHostDedupMatrix
 *
 * @brief Class for matrix duplication on host.
 *
 * @details Here is a basic example.
 *
 * referenced host matrix as input:
 * | Re(A)  Im(A)  Re(B)  Im(B) |
 * | Re(C)  Im(C)  Re(D)  Im(D) |
 *
 * output:
 * | Re(A)  Im(A)  Re(B)  Im(B) |
 * | Re(C)  Im(C)  Re(D)  Im(D) |
 * | Im(A) -Re(A)  Im(B) -Re(B) | <-- swap pair of Re and Im at this half of rows
 * | Im(C) -Re(C)  Im(D) -Re(D) | <-- and minus Re at this half of rows
 */
class CudaHostDedupMatrix : public CudaHostMatrix
{

public:
    // inherit constructors
    using CudaHostMatrix::CudaHostMatrix;

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
    __host__ RealSample
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        int32_t swap = 1 - 2 * ( col % 2 );
        int32_t DedupSign = ( row < height ) ? 1 : swap; // minus at Re(.)
        // int32_t DedupSign = ( row < height ) ? 1 : -swap; // minus at Im(.)

        return DedupSign * CudaHostMatrix::operator()( row % height, ( row < height ) ? col : col + swap );
    }

    /**
     * @brief Member function that returns the height of the matrix duplication object.
     *
     * @return uint32_t (height)
     */
    __host__ uint32_t getHeight() const
    {
        return height * 2;
    }

    uint32_t getNumElements() const
    {
        return getHeight() * getWidth();
    }

    /**
     * @brief Member function toDevice() with no change of objects check.
     *
     * @return const CudaDeviceDedupMatrix
     */
    const CudaDeviceDedupMatrix toDevice() const
    {
        return CudaDeviceDedupMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }

    void copyToMemory( RealSample* dst, cudaStream_t stream, bool transpose = false ) const;

    RealSample* allocateAndCopyToMemory( cudaStream_t stream, bool transpose = false ) const;
};

/**
 * @class CudaHostRingBuffer
 *
 * @brief Class for ringbuffer on host.
 */
class CudaHostRingBuffer : public CudaHostMatrix
{

public:
    /**
     * @brief A constructor for a new empty CudaHostRingBuffer object.
     */
    CudaHostRingBuffer() { }

    /**
     * @brief A constructor for a new CudaHostRingBuffer object with defined capacity.
     */
    CudaHostRingBuffer( uint32_t windowHeight, uint32_t width = 1, uint32_t heightCapacity = 0 )
        : CudaHostMatrix( heightCapacity == 0 ? windowHeight : heightCapacity, width )
        , offset( 0 )
        , usedHeight( 0 )
        , windowHeight( windowHeight )
    {
    }

    /**
     * @brief A constructor for a new CudaHostRingBuffer object with given matrix buffer pointer and capacity.
     */
    CudaHostRingBuffer( const CudaHostMatrix& other, uint32_t heightCapacity = 0 ); // TODO: change to windowHeight

    /**
     * @brief Returns the height (number of rows).
     *
     * @return uint32_t (height)
     */
    uint32_t getUsedHeight() const
    {
        return usedHeight > windowHeight ? windowHeight : usedHeight;
    }

    bool operator==( const CudaHostRingBuffer& other ) const;

    void pushRowFromMatrix( cudaStream_t stream, const CudaHostMatrix& m, uint32_t col );

    void pushRowsFromMatrix( cudaStream_t stream, const CudaHostMatrix& m );

    void moveWindow( uint32_t count = 2 );

    void pushEmptyRow( cudaStream_t stream );

    /**
     * @brief Return a reference to a single element on a host ringbuffer.
     *
     * @param[in] row
     * @param[in] col
     * @return reference to element
     */
    ThrustRealSampleDeviceVector::reference operator()( uint32_t row, uint32_t col = 0 )
    {
        assert( row < usedHeight );
        assert( row < windowHeight );
        assert( col < width );
        return CudaHostMatrix::operator()( ( row + offset ) % height, col );
    }

    /**
     * @brief Return a constant reference to a single element on a host ringbuffer.
     *
     * @param[in] row
     * @param[in] col
     * @return const_reference to element
     */
    ThrustRealSampleDeviceVector::const_reference operator()( uint32_t row, uint32_t col = 0 ) const
    {
        assert( row < usedHeight );
        assert( row < windowHeight );
        assert( col < width );
        return CudaHostMatrix::operator()( ( row + offset ) % height, col );
    }

    /**
     * @brief Member function toDevice().
     *
     * @return const CudaDeviceRingBuffer
     */
    CudaDeviceRingBuffer toDevice()
    {
        return CudaDeviceRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }

    /**
     * @brief Member function toDevice() with no change of objects check.
     *
     * @return const CudaDeviceRingBuffer
     */
    const CudaDeviceRingBuffer toDevice() const
    {
        return CudaDeviceRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }

    // allow access to private or protected information
    friend std::ostream& operator<<( std::ostream& os, const CudaHostRingBuffer& m );

protected:
    // protected variables
    uint32_t offset; ///< Offset
    uint32_t usedHeight; ///< Used height
    uint32_t windowHeight; ///< Window height
};

/*
 * @brief Output stream operator for debugging.
 *
 * @ detail
 * For debugging of paramset use this example:
 * std::cout << "CudaHostRingBuffer = " << m << std::endl;
 */
std::ostream& operator<<( std::ostream& os, const CudaHostRingBuffer& m );

/**
 * @class CudaHostDedupRingBuffer
 *
 * @brief Class for ringbuffer duplication on host.
 */
class CudaHostDedupRingBuffer : public CudaHostRingBuffer
{
public:
    using CudaHostRingBuffer::CudaHostRingBuffer; // inherit constructors

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
    __host__ RealSample
    operator()( uint32_t row, uint32_t col = 0 ) const
    {
        int32_t swap = 1 - 2 * ( row % 2 );
        int32_t DedupSign = ( col < width ) ? 1 : swap; // minus at Re(.)
        // int32_t DedupSign = ( col < width ) ? 1 : -swap; // minus at Im(.)

        return DedupSign * CudaHostRingBuffer::operator()( ( col < width ) ? row : row + swap, col % width );
    }

    /**
     * @brief Member function that returns the width of the ringbuffer duplication object.
     *
     * @return uint32_t (width)
     */
    __host__ uint32_t getWidth() const
    {
        return width * 2;
    }

    /**
     * @brief Member function toDevice() with no change of objects check.
     *
     * @return const CudaDeviceDedupRingBuffer
     */
    const CudaDeviceDedupRingBuffer toDevice() const
    {
        return CudaDeviceDedupRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }
};

/**
 * @class HostTrainingState
 *
 * @brief Class for Training state on host.
 */
class HostTrainingState
{

public:
    /**
     * @brief A constructor for a new empty HostTrainingState object.
     */
    HostTrainingState() { }

    /**
     * @brief A constructor for a new HostTrainingState object with defined capacity.
     */
    HostTrainingState( uint32_t linearLength, uint32_t dictWindowHeight, uint32_t dictCapacity = 0 )
        : basis( dictWindowHeight, linearLength / 2, dictCapacity )
        , gaussianCoeffs( dictWindowHeight, 1, dictCapacity )
        , linearCoeffs( linearLength )
    {
    }

    /**
     * @brief A constructor for a new HostTrainingState object with given ringbuffer content for basis and coefficients.
     */
    HostTrainingState( CudaHostDedupRingBuffer& _basis, CudaHostRingBuffer& _gaussianCoeffs, CudaHostMatrix& _linearCoeffs )
        : basis( _basis )
        , gaussianCoeffs( _gaussianCoeffs )
        , linearCoeffs( _linearCoeffs )
    {
    }

    /**
     * @brief Member function toDevice().
     *
     * @return DeviceTrainingState
     */
    DeviceTrainingState toDevice()
    {
        return DeviceTrainingState( basis.toDevice(), gaussianCoeffs.toDevice(), linearCoeffs.toDevice() );
    }

    /**
     * @brief Member function toDevice() with no change of objects check.
     *
     * @return const DeviceTrainingState
     */
    const DeviceTrainingState toDevice() const
    {
        return DeviceTrainingState( basis.toDevice(), gaussianCoeffs.toDevice(), linearCoeffs.toDevice() );
    }

    // public variables
    CudaHostDedupRingBuffer basis; ///< Basis ringbuffer
    CudaHostRingBuffer gaussianCoeffs; ///< Gaussian coefficient ringbuffer
    CudaHostMatrix linearCoeffs; ///< Linear coefficient ringbuffer
};

/**
 * @}
 */
