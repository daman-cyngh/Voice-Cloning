

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope



def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer):
	
	alignments, next_attention_state = attention_mechanism(
		cell_output, state=attention_state)

	
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
	
	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable(
		"attention_variable_projection", shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		"attention_bias", shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])

def _smoothing_normalization(e):
	
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):
	

	def __init__(self,
				 num_units,
				 memory,
				 hparams,
				 mask_encoder=True,
				 memory_sequence_length=None,
				 smoothing=False,
				 cumulate_weights=True,
				 name="LocationSensitiveAttention"):
		
		normalization_function = _smoothing_normalization if (smoothing == True) else None
		memory_length = memory_sequence_length if (mask_encoder==True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=memory_length,
				probability_fn=normalization_function,
				name=name)

		self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
			kernel_size=hparams.attention_kernel, padding="same", use_bias=True,
			bias_initializer=tf.zeros_initializer(), name="location_features_convolution")
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
			dtype=tf.float32, name="location_features_layer")
		self._cumulate = cumulate_weights

	def __call__(self, query, state):
		
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			
			processed_query = self.query_layer(query) if self.query_layer else query
			
			processed_query = tf.expand_dims(processed_query, 1)

			
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			
			f = self.location_convolution(expanded_alignments)
			
			processed_location_features = self.location_layer(f)

			
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)


		
		alignments = self._probability_fn(energy, previous_alignments)

		
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments

		return alignments, next_state
