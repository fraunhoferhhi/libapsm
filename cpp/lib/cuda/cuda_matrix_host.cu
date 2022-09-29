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
 * @file cuda_matrix_host.cu
 * @brief APSM matrix multiplication
 *
 * @author Daniel Schäufele, HHI,
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.10.29   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 * @date 2020.07.17   0.03    refactoring
 *
 * @note GITHUB THRUST - A Parallel Algorithms Library
 *       https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
 */

// THRUST
#include <thrust/copy.h>

// APSM
#include "cuda/cuda_errorhandling.cuh"
#include "cuda/cuda_indexing.cuh"
#include "cuda/cuda_matrix_device.cuh"
#include "cuda/cuda_matrix_functor.cuh"
#include "cuda/cuda_matrix_host.cuh"

/**
 * @brief Construct a new CudaMatrix with given dimensions, that is filled with the given value (0 by default).
 *
 * @param _height Number of rows
 * @param _width Number of columns
 * @param fill_value Value that will be used to fill the matrix
 */
CudaHostMatrix::CudaHostMatrix( uint32_t _height, uint32_t _width, RealSample fill_value )
    : height( _height )
    , width( _width )
    , elements( width * height, fill_value )
{
}

/**
 * @brief Construct a new CudaMatrix with dimensions and entries given by a ComplexSampleMatrix.
 *
 * Depending on the complexConversion parameter, the complex values are mapped to the real values in a different way. As an example, the complex matrix
 *     | A B |
 *     | C D |
 * - with the parameter `R_I` will be mapped to
 *     | Re(A) Re(B) |
 *     | Re(C) Re(D) |
 *     | Im(A) Im(B) |
 *     | Im(C) Im(D) |
 * - with the parameter `RI_ImR` will be mapped to
 *     | Re(A)  Im(A) Re(B)  Im(B) |
 *     | Re(C)  Im(C) Re(D)  Im(D) |
 *     | Im(A) -Re(A) Im(B) -Re(B) |
 *     | Im(C) -Re(C) Im(D) -Re(D) |
 * - with the parameter `RI` will be mapped to
 *     | Re(A) Im(A) Re(B) Im(B) |
 *     | Re(C) Im(C) Re(D) Im(D) |
 *
 * @param data Complex data that will be used to fill the matrix
 * @param complexConversion Pattern that will be used to map complex values to real values
 */
CudaHostMatrix::CudaHostMatrix( cudaStream_t stream, const ComplexSampleMatrix& data )
    : height( data.size() )
    , width( data[ 0 ].size() * 2 )
    , elements( width * height )
{
    uint32_t data_height = data.size();
    uint32_t data_width = data[ 0 ].size();
    ThrustComplexSampleDeviceVector d_data( data_width * data_height );

    for ( uint32_t idx = 0; idx < data_height; idx++ )
    {
        thrust::copy( /*thrust::cuda::par.on( s1 ),*/ data[ idx ].begin(), data[ idx ].end(), d_data.begin() + ( idx * data_width ) );
    }

    typedef ThrustRealSampleDeviceVector::iterator Iterator;

    strided_range<Iterator> evens( elements.begin(), elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), evens.begin(), functor_real() );

    strided_range<Iterator> odds( elements.begin() + 1, elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), odds.begin(), functor_imag() );
}

/**
 * @brief Construct a new CudaMatrix with dimensions and entries given by a ComplexSampleMatrix.
 */
CudaHostMatrix::CudaHostMatrix( cudaStream_t stream, const ComplexSampleVector& data )
    : height( 1 )
    , width( data.size() * 2 )
    , elements( width * height )
{
    uint32_t data_height = 1;
    uint32_t data_width = data.size();
    ThrustComplexSampleDeviceVector d_data( data_width * data_height );

    thrust::copy( /*thrust::cuda::par.on( s1 ),*/ data.begin(), data.end(), d_data.begin() );

    typedef ThrustRealSampleDeviceVector::iterator Iterator;

    strided_range<Iterator> evens( elements.begin(), elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), evens.begin(), functor_real() );

    strided_range<Iterator> odds( elements.begin() + 1, elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), odds.begin(), functor_imag() );
}

/**
 * @brief Construct a new CudaMatrix with dimension and entries given by a RealSampleMatrix.
 *
 * @param data Real data that will be used to fill the matrix
 */
CudaHostMatrix::CudaHostMatrix( const RealSampleMatrix& data )
    : height( data.size() )
    , width( data[ 0 ].size() )
    , elements( width * height )
{
    for ( uint32_t idx = 0; idx < height; idx++ )
    {
        thrust::copy( /*thrust::cuda::par.on( s1 ),*/ data[ idx ].begin(), data[ idx ].end(), elements.begin() + ( idx * width ) );
    }
}

/**
 * @brief Returns a vector of complex values from a CudaMatrix given in `RI` format (see constructor above for definitions).
 *
 * @return Complex representation of CudaMatrix
 */
ThrustComplexSampleDeviceVector CudaHostMatrix::toComplexVector( cudaStream_t stream )
{
    uint32_t num_complex_values = width * height / 2;
    ThrustComplexSampleDeviceVector out( num_complex_values );

    typedef ThrustRealSampleDeviceVector::iterator Iterator;
    strided_range<Iterator> evens( elements.begin(), elements.end(), 2 );
    strided_range<Iterator> odds( elements.begin() + 1, elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), evens.begin(), evens.end(), odds.begin(), out.begin(), functor_complex() );

    return out;
}

/**
 * @brief Compares CudaHostMatrix to another CudaHostMatrix.
 *
 * @param other CudaHostMatrix to compare to
 * @return true if dimensions and all entries are equal
 * @return false otherwise
 */
bool CudaHostMatrix::operator==( const CudaHostMatrix& other ) const
{
    if ( height != other.height || width != other.width )
        return false;

    // create a CUDA stream
    cudaStream_t s1;
    cudaStreamCreate( &s1 );

    bool result = thrust::equal( thrust::cuda::par.on( s1 ), elements.begin(), elements.end(), other.elements.begin() );

    // destroy streams
    cudaStreamDestroy( s1 );

    return result;
}

void CudaHostMatrix::copyToMemory( RealSample* dst, cudaStream_t stream ) const
{
    CUDA_CHECK( cudaMemcpyAsync( dst, getRawDataPointer(), getNumElements() * sizeof( RealSample ), cudaMemcpyDeviceToDevice, stream ) );
}

RealSample* CudaHostMatrix::allocateAndCopyToMemory( cudaStream_t stream ) const
{
    float* dst = 0;
    CUDA_CHECK( cudaMalloc( &dst, getNumElements() * sizeof( RealSample ) ) );
    copyToMemory( dst, stream );
    return dst;
}

/**
 * @brief Prints a human readable representation of CudaHostMatrix to a stream.
 *
 * @param os Stream to print to
 * @param m CudaHostMatrix that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const CudaHostMatrix& m )
{
    os << "[" << std::endl;
    for ( uint32_t idx = 0; idx < m.height; idx++ )
    {
        os << "  [";
        for ( uint32_t jdx = 0; jdx < m.width; jdx++ )
        {
            os << m( idx, jdx );
            if ( jdx < m.width - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}

__global__ void kernelCopyToMemory( float* dst, const CudaDeviceDedupMatrix m, bool transpose )
{
    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();
    const uint32_t batchSize = getBlockDim_1D();

    for ( uint32_t idx = blockThreadId; idx < m.getHeight(); idx += batchSize )
        if ( transpose )
            dst[ IDX2E( idx, blockId, m.getHeight() ) ] = m( idx, blockId );
        else
            dst[ IDX2E( blockId, idx, m.getWidth() ) ] = m( idx, blockId );
}

void CudaHostDedupMatrix::copyToMemory( RealSample* dst, cudaStream_t stream, bool transpose ) const
{
    CudaDeviceDedupMatrix deviceMatrix = toDevice();

    dim3 blockDim, gridDim;
    apsm_kernel_dims( &blockDim, &gridDim, getWidth() );

    kernelCopyToMemory<<<gridDim, blockDim, 0, stream>>>( dst, deviceMatrix, transpose );
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
}

RealSample* CudaHostDedupMatrix::allocateAndCopyToMemory( cudaStream_t stream, bool transpose ) const
{
    float* dst = 0;
    CUDA_CHECK( cudaMalloc( &dst, getNumElements() * sizeof( RealSample ) ) );
    copyToMemory( dst, stream, transpose );
    return dst;
}

CudaHostRingBuffer::CudaHostRingBuffer( const CudaHostMatrix& m, uint32_t heightCapacity )
    : CudaHostMatrix( heightCapacity == 0 ? m.getHeight() : heightCapacity, m.getWidth() )
    , offset( 0 )
    , usedHeight( std::min( m.getHeight(), getHeight() ) )
{
    uint32_t copy_offset = m.getHeight() - getHeight();
    thrust::copy( m.elements.begin() + copy_offset * getWidth(), m.elements.end(), elements.begin() );
}

void CudaHostRingBuffer::pushRowFromMatrix( cudaStream_t stream, const CudaHostMatrix& m, uint32_t col )
{
    uint32_t nextRow;

    if ( usedHeight < height )
    {
        nextRow = usedHeight;
        usedHeight++;
    }
    else
    {
        nextRow = offset;
        offset = ( offset + 1 ) % height;
    }

    // ugly hack, because strided_range doesn't work with const objects
    CudaHostMatrix& m_nonconst = const_cast<CudaHostMatrix&>( m );
    typedef ThrustRealSampleDeviceVector::iterator Iterator;
    strided_range<Iterator> column_iterator( m_nonconst.elements.begin() + col, m_nonconst.elements.end(), m_nonconst.width );
    thrust::copy( thrust::cuda::par.on( stream ), column_iterator.begin(), column_iterator.end(), elements.begin() + nextRow * width );
}

/**
 * @brief CUDA dictionary kernel
 * @details Copy a transposed version of the rx_data matrix into the basis matrix.
 *
 * @param[out] basis basis matrix (dictionary)
 * @param[in] rx_data data constellations
 *
 * @return void
 */
__global__ void kernel_apsm_dictionary( CudaDeviceRingBuffer basis, const CudaDeviceMatrix rx_data, uint32_t oldUsedHeight )
{
    const uint32_t blockId = getBlockIdx_1D();
    const uint32_t blockThreadId = getBlockThreadIdx_1D();

    const uint32_t batch_size = getBlockDim_1D();

    const uint32_t linearLength = rx_data.getHeight();

    for ( uint32_t batch_idx = blockThreadId; batch_idx < linearLength; batch_idx += batch_size )
    {
        basis( oldUsedHeight + blockId, batch_idx ) = rx_data( batch_idx, blockId );
    }
}

void CudaHostRingBuffer::pushRowsFromMatrix( cudaStream_t stream, const CudaHostMatrix& m )
{
    // temporarily raise usedHeight to allow writing to whole matrix
    uint32_t oldUsedHeight = usedHeight;
    usedHeight += m.getWidth();

    uint32_t vector_size = m.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate shared memory size parameter
    uint32_t sharedMemorySize = 0;

    // run kernel
    kernel_apsm_dictionary<<<grid_dim, block_dim, sharedMemorySize, stream>>>( this->toDevice(), m.toDevice(), oldUsedHeight );

    usedHeight = oldUsedHeight;
}

void CudaHostRingBuffer::moveWindow( uint32_t count )
{
    if ( usedHeight < height )
    {
        usedHeight += count;
    }
    else
    {
        offset = ( offset + count ) % height;
    }
}

void CudaHostRingBuffer::pushEmptyRow( cudaStream_t stream )
{
    if ( usedHeight < height )
    {
        usedHeight++;
    }
    else
    {
        thrust::fill( thrust::cuda::par.on( stream ), elements.begin() + offset * width, elements.begin() + ( offset + 1 ) * width, 0 );
        offset = ( offset + 1 ) % height;
    }
}

/**
 * @brief Compares CudaHostRingBuffer to another CudaHostRingBuffer.
 *
 * @param other CudaHostRingBuffer to compare to
 * @return true if dimensions and all entries are equal
 * @return false otherwise
 */
bool CudaHostRingBuffer::operator==( const CudaHostRingBuffer& other ) const
{
    if ( usedHeight != other.usedHeight || width != other.width )
        return false;

    for ( uint32_t idx = 0; idx < usedHeight; idx++ )
        for ( uint32_t jdx = 0; jdx < width; jdx++ )
            if ( this->operator()( idx, jdx ) != other( idx, jdx ) )
                return false;

    return true;
}

/**
 * @brief Prints a human readable representation of CudaHostRingBuffer to a stream.
 *
 * @param os Stream to print to
 * @param m CudaHostRingBuffer that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
std::ostream& operator<<( std::ostream& os, const CudaHostRingBuffer& m )
{
    os << "[" << std::endl;
    for ( uint32_t idx = 0; idx < m.usedHeight; idx++ )
    {
        os << "  [";
        for ( uint32_t jdx = 0; jdx < m.width; jdx++ )
        {
            os << m( idx, jdx );
            if ( jdx < m.width - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}
