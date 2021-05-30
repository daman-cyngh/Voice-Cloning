import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


class TacoTestHelper(Helper):
	def __init__(self, batch_size, hparams):
		with tf.name_scope("TacoTestHelper"):
			self._batch_size = batch_size
			self._output_dim = hparams.num_mels
			self._reduction_factor = hparams.outputs_per_step
			self.stop_at_any = hparams.stop_at_any

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		
		with tf.name_scope("TacoTestHelper"):
			
			finished = tf.cast(tf.round(stop_token_prediction), tf.bool)

			
			if self.stop_at_any:
				finished = tf.reduce_any(tf.reduce_all(finished, axis=0)) 
			else:
				finished = tf.reduce_all(tf.reduce_all(finished, axis=0)) 

			
			next_inputs = outputs[:, -self._output_dim:]
			next_state = state
			return (finished, next_inputs, next_state)


class TacoTrainingHelper(Helper):
	def __init__(self, batch_size, targets, hparams, gta, evaluating, global_step):
		
		with tf.name_scope("TacoTrainingHelper"):
			self._batch_size = batch_size
			self._output_dim = hparams.num_mels
			self._reduction_factor = hparams.outputs_per_step
			self._ratio = tf.convert_to_tensor(hparams.tacotron_teacher_forcing_ratio)
			self.gta = gta
			self.eval = evaluating
			self._hparams = hparams
			self.global_step = global_step

			r = self._reduction_factor
			
			self._targets = targets[:, r-1::r, :]

			
			self._lengths = tf.tile([tf.shape(self._targets)[1]], [self._batch_size])

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def token_output_size(self):
		return self._reduction_factor

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return np.int32

	def initialize(self, name=None):
		
		if self.gta:
			self._ratio = tf.convert_to_tensor(1.) 
		elif self.eval and self._hparams.natural_eval:
			self._ratio = tf.convert_to_tensor(0.) 
		else:
			if self._hparams.tacotron_teacher_forcing_mode == "scheduled":
				self._ratio = _teacher_forcing_ratio_decay(self._hparams.tacotron_teacher_forcing_init_ratio,
					self.global_step, self._hparams)

		return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

	def sample(self, time, outputs, state, name=None):
		return tf.tile([0], [self._batch_size])  

	def next_inputs(self, time, outputs, state, sample_ids, stop_token_prediction, name=None):
		with tf.name_scope(name or "TacoTrainingHelper"):
			
			finished = (time + 1 >= self._lengths)

			
			next_inputs = tf.cond(
				tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
				lambda: self._targets[:, time, :], 
				lambda: outputs[:,-self._output_dim:])

			
			next_state = state
			return (finished, next_inputs, next_state)


def _go_frames(batch_size, output_dim):
	
	return tf.tile([[0.0]], [batch_size, output_dim])

def _teacher_forcing_ratio_decay(init_tfr, global_step, hparams):
		
		tfr = tf.train.cosine_decay(init_tfr,
			global_step=global_step - hparams.tacotron_teacher_forcing_start_decay, 
			decay_steps=hparams.tacotron_teacher_forcing_decay_steps, 
			alpha=hparams.tacotron_teacher_forcing_decay_alpha, 
			name="tfr_cosine_decay")

		
		narrow_tfr = tf.cond(
			tf.less(global_step, tf.convert_to_tensor(hparams.tacotron_teacher_forcing_start_decay)),
			lambda: tf.convert_to_tensor(init_tfr),
			lambda: tfr)

		return narrow_tfr