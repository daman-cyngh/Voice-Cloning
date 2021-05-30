
import collections
import tensorflow as tf
from synthesizer.models.attention import _compute_attention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest

_zero_state_tensors = rnn_cell_impl._zero_state_tensors



class TacotronEncoderCell(RNNCell):
	

	def __init__(self, convolutional_layers, lstm_layer):
		
		super(TacotronEncoderCell, self).__init__()
		
		self._convolutions = convolutional_layers
		self._cell = lstm_layer

	def __call__(self, inputs, input_lengths=None):
		
		conv_output = self._convolutions(inputs)

		
		hidden_representation = self._cell(conv_output, input_lengths)

		
		self.conv_output_shape = conv_output.shape
		return hidden_representation


class TacotronDecoderCellState(
	collections.namedtuple("TacotronDecoderCellState",
	 ("cell_state", "attention", "time", "alignments",
	  "alignment_history"))):
	
	def replace(self, **kwargs):
		
		return super(TacotronDecoderCellState, self)._replace(**kwargs)

class TacotronDecoderCell(RNNCell):
	

	def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection):
		
		super(TacotronDecoderCell, self).__init__()
		
		self._prenet = prenet
		self._attention_mechanism = attention_mechanism
		self._cell = rnn_cell
		self._frame_projection = frame_projection
		self._stop_projection = stop_projection

		self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

	def _batch_size_checks(self, batch_size, error_message):
		return [check_ops.assert_equal(batch_size,
		  self._attention_mechanism.batch_size,
		  message=error_message)]

	@property
	def output_size(self):
		return self._frame_projection.shape

	@property
	def state_size(self):
		
		return TacotronDecoderCellState(
			cell_state=self._cell._cell.state_size,
			time=tensor_shape.TensorShape([]),
			attention=self._attention_layer_size,
			alignments=self._attention_mechanism.alignments_size,
			alignment_history=())

	def zero_state(self, batch_size, dtype):
		
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			cell_state = self._cell._cell.zero_state(batch_size, dtype)
			error_message = (
				"When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the requested batch size.")
			with ops.control_dependencies(
				self._batch_size_checks(batch_size, error_message)):
				cell_state = nest.map_structure(
					lambda s: array_ops.identity(s, name="checked_cell_state"),
					cell_state)
			return TacotronDecoderCellState(
				cell_state=cell_state,
				time=array_ops.zeros([], dtype=tf.int32),
				attention=_zero_state_tensors(self._attention_layer_size, batch_size,
				  dtype),
				alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
				alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
				dynamic_size=True))

	def __call__(self, inputs, state):
		
		prenet_output = self._prenet(inputs)

		
		LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

		
		LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)


		
		previous_alignments = state.alignments
		previous_alignment_history = state.alignment_history
		context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
			LSTM_output,
			previous_alignments,
			attention_layer=None)

		
		projections_input = tf.concat([LSTM_output, context_vector], axis=-1)

		
		cell_outputs = self._frame_projection(projections_input)
		stop_tokens = self._stop_projection(projections_input)

		
		alignment_history = previous_alignment_history.write(state.time, alignments)

		
		next_state = TacotronDecoderCellState(
			time=state.time + 1,
			cell_state=next_cell_state,
			attention=context_vector,
			alignments=cumulated_alignments,
			alignment_history=alignment_history)

		return (cell_outputs, stop_tokens), next_state
