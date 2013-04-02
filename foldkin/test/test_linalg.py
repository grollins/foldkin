import nose.tools
import numpy
import scipy.linalg
from foldkin.probability_vector import make_prob_vec_from_state_ids,\
                                        make_prob_vec_from_panda_series
from foldkin.rate_matrix import make_rate_matrix_from_state_ids
from foldkin.linalg import vector_product, vector_matrix_product,\
                            asym_vector_matrix_product,\
                            ScipyMatrixExponential

@nose.tools.istest
def computes_vector_product_with_ordered_indices():
    state_ids = ['a', 'b', 'c']
    prob_vec1 = make_prob_vec_from_state_ids(state_ids)
    prob_vec2 = make_prob_vec_from_state_ids(state_ids)
    prob_vec1.set_state_probability('a', 0.1)
    prob_vec1.set_state_probability('b', 0.3)
    prob_vec1.set_state_probability('c', 0.6)
    prob_vec2.set_state_probability('a', 0.1)
    prob_vec2.set_state_probability('b', 0.3)
    prob_vec2.set_state_probability('c', 0.6)
    prob_vec_product = vector_product(prob_vec1, prob_vec2, do_alignment=False)

    numpy_vec1 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec2 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec_product = numpy.dot(numpy_vec1, numpy_vec2.T)

    error_msg = "Expected %.2f, got %.2f" % (numpy_vec_product, prob_vec_product)
    nose.tools.eq_(numpy_vec_product, prob_vec_product, error_msg)

@nose.tools.istest
def computes_vector_product_with_unordered_indices():
    state_ids = ['a', 'b', 'c']
    unordered_state_ids = ['b', 'c', 'a']
    prob_vec1 = make_prob_vec_from_state_ids(state_ids)
    prob_vec2 = make_prob_vec_from_state_ids(unordered_state_ids)
    prob_vec1.set_state_probability('a', 0.1)
    prob_vec1.set_state_probability('b', 0.3)
    prob_vec1.set_state_probability('c', 0.6)
    prob_vec2.set_state_probability('a', 0.1)
    prob_vec2.set_state_probability('b', 0.3)
    prob_vec2.set_state_probability('c', 0.6)
    prob_vec_product = vector_product(prob_vec1, prob_vec2, do_alignment=True)

    numpy_vec1 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec2 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec_product = numpy.dot(numpy_vec1, numpy_vec2.T)

    error_msg = "Expected %.2f, got %.2f" % (numpy_vec_product, prob_vec_product)
    nose.tools.eq_(numpy_vec_product, prob_vec_product, error_msg)

@nose.tools.istest
def computes_vector_matrix_with_only_one_entry_and_output_type_is_series():
    state_ids = ['a']
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'a', 1.5)
    product_vec = vector_matrix_product(prob_vec, rate_matrix,
                                         do_alignment=False)
    error_msg = "%s, %s" % (type(product_vec.series), type(prob_vec.series))
    nose.tools.ok_(type(product_vec.series) == type(prob_vec.series),
                   error_msg)

@nose.tools.istest
def compute_vector_matrix_product_with_ordered_indices():
    state_ids = ['a', 'b', 'c']
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'b', 0.5)
    rate_matrix.set_rate('a', 'c', 0.0)
    rate_matrix.set_rate('b', 'c', 0.1)
    rate_matrix.set_rate('b', 'a', 0.0)
    rate_matrix.set_rate('c', 'a', 0.9)
    rate_matrix.set_rate('c', 'b', 0.0)
    rate_matrix.balance_transition_rates()
    product_vec = vector_matrix_product(prob_vec, rate_matrix,
                                         do_alignment=False)
    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = -npy_array[0,:].sum()
    npy_array[1,1] = -npy_array[1,:].sum()
    npy_array[2,2] = -npy_array[2,:].sum()
    npy_product = numpy.dot(npy_vec, npy_array)

    nose.tools.eq_( npy_product[0,0], product_vec.get_state_probability('a') )
    nose.tools.eq_( npy_product[0,1], product_vec.get_state_probability('b') )
    nose.tools.eq_( npy_product[0,2], product_vec.get_state_probability('c') )

@nose.tools.istest
def compute_vector_matrix_product_with_unordered_indices():
    state_ids = ['a', 'b', 'c']
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    unordered_state_ids = ['b', 'c', 'a']
    rate_matrix = make_rate_matrix_from_state_ids(unordered_state_ids,
                                                  unordered_state_ids)
    rate_matrix.set_rate('a', 'b', 0.5)
    rate_matrix.set_rate('a', 'c', 0.0)
    rate_matrix.set_rate('b', 'c', 0.1)
    rate_matrix.set_rate('b', 'a', 0.0)
    rate_matrix.set_rate('c', 'a', 0.9)
    rate_matrix.set_rate('c', 'b', 0.0)
    rate_matrix.balance_transition_rates()
    product_vec = vector_matrix_product(prob_vec, rate_matrix,
                                        do_alignment=True)

    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = -npy_array[0,:].sum()
    npy_array[1,1] = -npy_array[1,:].sum()
    npy_array[2,2] = -npy_array[2,:].sum()
    npy_product = numpy.dot(npy_vec, npy_array)

    nose.tools.eq_( npy_product[0,0], product_vec.get_state_probability('a') )
    nose.tools.eq_( npy_product[0,1], product_vec.get_state_probability('b') )
    nose.tools.eq_( npy_product[0,2], product_vec.get_state_probability('c') )

@nose.tools.istest
def compute_asymmetric_vector_matrix_product_with_unordered_indices():
    state_ids = ['a', 'b', 'c']
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    unordered_state_ids = ['b', 'c', 'a']
    column_ids = ['d', 'f']
    rate_matrix = make_rate_matrix_from_state_ids(
                    unordered_state_ids, column_ids)
    rate_matrix.set_rate('a', 'd', 0.01)
    rate_matrix.set_rate('a', 'f', 0.05)
    rate_matrix.set_rate('b', 'd', 0.00)
    rate_matrix.set_rate('b', 'f', 0.10)
    rate_matrix.set_rate('c', 'd', 0.10)
    rate_matrix.set_rate('c', 'f', 0.00)
    product_vec = asym_vector_matrix_product(prob_vec, rate_matrix,
                                              do_alignment=True)
    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,2] )
    npy_array[0,0] = 0.01  # a,d
    npy_array[0,1] = 0.05  # a,f
    npy_array[1,0] = 0.00  # b,d
    npy_array[1,1] = 0.10  # b,f
    npy_array[2,0] = 0.10  # c,d
    npy_array[2,1] = 0.00  # c,f
    npy_product = numpy.dot(npy_vec, npy_array)
    nose.tools.eq_( npy_product[0,0], product_vec.get_state_probability('d') )
    nose.tools.eq_( npy_product[0,1], product_vec.get_state_probability('f') )

@nose.tools.istest
def compute_matrix_exponential():
    expm = ScipyMatrixExponential()
    state_ids = ['a', 'b', 'c']
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'b', 10.0)
    rate_matrix.set_rate('a', 'c', 0.1)
    rate_matrix.set_rate('b', 'c', 1.2)
    rate_matrix.set_rate('b', 'a', 0.01)
    rate_matrix.set_rate('c', 'a', 3.2)
    rate_matrix.set_rate('c', 'b', 0.2)
    rate_matrix.balance_transition_rates()
    expQt_matrix = expm.compute_matrix_exp(rate_matrix, dwell_time=0.1)

    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 10.0   # a,b
    npy_array[0,2] = 0.1    # a,c
    npy_array[1,2] = 1.2    # b,c
    npy_array[1,0] = 0.01   # b,a
    npy_array[2,0] = 3.2    # c,a
    npy_array[2,1] = 0.2    # c,b
    npy_array[0,0] = -npy_array[0,:].sum()
    npy_array[1,1] = -npy_array[1,:].sum()
    npy_array[2,2] = -npy_array[2,:].sum()
    scipy_product_matrix = scipy.linalg.expm(npy_array * 0.1)

    print expQt_matrix
    print scipy_product_matrix
    nose.tools.ok_(numpy.allclose(scipy_product_matrix,
                                  expQt_matrix.as_npy_array()))

@nose.tools.istest
def compute_matrix_exponential_with_large_rates():
    expm = ScipyMatrixExponential()
    state_ids = ['a', 'b', 'c']
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'b', 1e6)
    rate_matrix.set_rate('a', 'c', 1e0)
    rate_matrix.set_rate('b', 'c', 1e1)
    rate_matrix.set_rate('b', 'a', 1e-2)
    rate_matrix.set_rate('c', 'a', 1e1)
    rate_matrix.set_rate('c', 'b', 1e3)
    rate_matrix.balance_transition_rates()
    expQt_matrix = expm.compute_matrix_exp(rate_matrix, dwell_time=0.1)

    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 1e6   # a,b
    npy_array[0,2] = 1e0   # a,c
    npy_array[1,2] = 1e1   # b,c
    npy_array[1,0] = 1e-2   # b,a
    npy_array[2,0] = 1e1   # c,a
    npy_array[2,1] = 1e3   # c,b
    npy_array[0,0] = -npy_array[0,:].sum()
    npy_array[1,1] = -npy_array[1,:].sum()
    npy_array[2,2] = -npy_array[2,:].sum()
    scipy_product_matrix = scipy.linalg.expm(npy_array * 0.1)

    print expQt_matrix
    print scipy_product_matrix
    nose.tools.ok_(numpy.allclose(scipy_product_matrix,
                                  expQt_matrix.as_npy_array()))
