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
 * @file cuda_matrix_functor.cuh
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

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

/**
 * @brief helper class that iterates over every n-th element
 *
 * @tparam Iterator
 */
template <typename Iterator>
class strided_range
{
public:
    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type, difference_type>
    {
        difference_type stride;

        stride_functor( difference_type stride )
            : stride( stride )
        {
        }

        __device__ difference_type operator()( const difference_type& dt ) const
        {
            return stride * dt;
        }
    };

    typedef typename thrust::counting_iterator<difference_type> CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

    // construct strided_range for the range [first,last)
    strided_range( Iterator first, Iterator last, difference_type stride )
        : first( first )
        , last( last )
        , stride( stride )
    {
    }

    PermutationIterator begin( void ) const
    {
        return PermutationIterator( first, TransformIterator( CountingIterator( 0 ), stride_functor( stride ) ) );
    }

    PermutationIterator end( void ) const
    {
        return begin() + ( ( last - first ) + ( stride - 1 ) ) / stride;
    }

protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

/**
 * @brief Functor that extracts the real part of a complex input
 */
struct functor_real
{

    /**
     * @brief extracts positive real part from a complex value
     *
     * @param[in] in complex value
     * @return real part
     */
    __device__ RealSample operator()( ThrustComplexSample& in )
    {
        return in.real();
    }
};

/**
 * @brief Functor that extracts the imaginary part of a complex input
 */
struct functor_imag
{

    /**
     * @brief extracts positive imaginary part from a complex value
     *
     * @param[in] in complex value
     * @return imaginary part
     */
    __device__ RealSample operator()( ThrustComplexSample& in )
    {
        return in.imag();
    }
};

/**
 * @brief Functor that constructs complex objects out of real and imaginary input
 */
struct functor_complex
{

    /**
     * @brief combine real and imaginary part to a complex value
     *
     * @param[in] re value representing the real part
     * @param[in] im value representing the imaginary part
     * @return comlex value
     */
    __device__ ThrustComplexSample operator()( RealSample re, RealSample im )
    {
        return ThrustComplexSample( re, im );
    }
};

/**
 * @}
 */
