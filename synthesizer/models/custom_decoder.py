from __future__ import absolute_import, division, print_function
import collections
import tensorflow as tf
from synthesizer.models.helpers import TacoTestHelper, TacoTrainingHelper
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


class CustomDecoderOutput(
		collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
	pass


class CustomDecoder(decoder.Decoder):
	

	def __init__(self, cell, helper, initial_state, output_layer=None):
		
		rnn_cell_impl.assert_like_rnncell(type(cell), cell)
		if not isinstance(helper, helper_py.Helper):
			raise TypeError("helper must be a Helper, received: %s" % type(helper))
		if (output_layer is not None
				and not isinstance(output_layer, layers_base.Layer)):
			raise TypeError(
					"output_layer must be a Layer, received: %s" % type(output_layer))
		self._cell = cell
		self._helper = helper
		self._initial_state = initial_state
		self._output_layer = output_layer

	@property
	def batch_size(self):
		return self._helper.batch_size

	def _rnn_output_size(self):
		size = self._cell.output_size
		if self._output_layer is None:
			return size
		else:
			
			output_shape_with_unknown_batch = nest.map_structure(
					lambda s: tensor_shape.TensorShape([None]).concatenate(s),
					size)
			layer_output_shape = self._output_layer._compute_output_shape(  
					output_shape_with_unknown_batch)
			return nest.map_structure(lambda s: s[1:], layer_output_shape)

	@property
	def output_size(self):
		
		return CustomDecoderOutput(
				rnn_output=self._rnn_output_size(),
				token_output=self._helper.token_output_size,
				sample_id=self._helper.sample_ids_shape)

	@property
	def output_dtype(self):
		
		dtype = nest.flatten(self._initial_state)[0].dtype
		return CustomDecoderOutput(
				nest.map_structure(lambda _: dtype, self._rnn_output_size()),
				tf.float32,
				self._helper.sample_ids_dtype)

	def initialize(self, name=None):
		
		return self._helper.initialize() + (self._initial_state,)

	def step(self, time, inputs, state, name=None):
		
		with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
			
			(cell_outputs, stop_token), cell_state = self._cell(inputs, state)

			
			if self._output_layer is not None:
				cell_outputs = self._output_layer(cell_outputs)
			sample_ids = self._helper.sample(
					time=time, outputs=cell_outputs, state=cell_state)

			(finished, next_inputs, next_state) = self._helper.next_inputs(
					time=time,
					outputs=cell_outputs,
					state=cell_state,
					sample_ids=sample_ids,
					stop_token_prediction=stop_token)

		outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
		return (outputs, next_state, next_inputs, finished)
