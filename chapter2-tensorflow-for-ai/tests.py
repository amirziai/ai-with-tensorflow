import numpy as np
import tensorflow as tf


def _success_message():
	return 'All tests passed'

def _run_test(f, actuals, expected, messages):
	for actual, exp, msg in zip(actuals, expected, messages):
		assert f(*actual) == exp, msg

def _evaluate_tensor(tensor):
	return tf.Session().run(tensor)

def test_get_tensor_shape(f):
	shape = [6, 2]
	tensor_shape = f(tf.constant(0, shape=shape))
	assert tensor_shape == shape
	return _success_message()

def test_get_tensor_rank(f):
	shape = [3, 7]
	rank = f(tf.constant(0, shape=shape))
	assert rank == len(shape)
	return _success_message()

def test_get_tensor_dtype(f):
	shape = [3, 7]
	value = 1.2
	dtype = f(tf.constant(value, shape=shape))
	assert dtype == tf.float32
	return _success_message()

def test_add(f):
	actuals = [(1, 0), (2, 1)]
	expected = [1, 3]
	messages = [
	'add(1, 0) should return 1',
	'add(2, 1) should return 3'
	]
	_run_test(f, actuals, expected, messages)
	return _success_message()

def test_add_rank0_tensors(f):
	x = 1
	y = 2
	z_tensor = f(x, y)
	z = tf.Session().run(z_tensor)
	assert z == x + y
	return _success_message()

def test_add_rank1_tensors(f):
	xs = [1, 2, 3]
	ys = [6, 5, 4]
	z_tensor = f(xs, ys)
	z = tf.Session().run(z_tensor)
	assert all(z == [x + y for x, y in zip(xs, ys)])
	return _success_message()

def test_create_constant_tensor(f):
	value = 42
	m = 5
	n = 3
	tensor = f(value, m, n)
	array_tf = _evaluate_tensor(tensor)
	array_np = np.full(shape=[m, n], fill_value=value)
	np.array_equal(array_tf, array_np)
	return _success_message()

def test_create_fill_tensor(f):
	value = 42
	m = 5
	n = 3
	tensor = f(value, m, n)
	array_tf = _evaluate_tensor(tensor)
	array_np = np.full(shape=[m, n], fill_value=value)
	np.array_equal(array_tf, array_np)
	return _success_message()
