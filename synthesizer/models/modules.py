import tensorflow as tf


class HighwayNet:
    def __init__(self, units, name=None):
        self.units = units
        self.scope = "HighwayNet" if name is None else name
        
        self.H_layer = tf.layers.Dense(units=self.units, activation=tf.nn.relu, name="H")
        self.T_layer = tf.layers.Dense(units=self.units, activation=tf.nn.sigmoid, name="T",
                                       bias_initializer=tf.constant_initializer(-1.))
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            H = self.H_layer(inputs)
            T = self.T_layer(inputs)
            return H * T + inputs * (1. - T)


class CBHG:
    def __init__(self, K, conv_channels, pool_size, projections, projection_kernel_size,
                 n_highwaynet_layers, highway_units, rnn_units, is_training, name=None):
        self.K = K
        self.conv_channels = conv_channels
        self.pool_size = pool_size
        
        self.projections = projections
        self.projection_kernel_size = projection_kernel_size
        
        self.is_training = is_training
        self.scope = "CBHG" if name is None else name
        
        self.highway_units = highway_units
        self.highwaynet_layers = [
            HighwayNet(highway_units, name="{}_highwaynet_{}".format(self.scope, i + 1)) for i in
            range(n_highwaynet_layers)]
        self._fw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name="{}_forward_RNN".format(self.scope))
        self._bw_cell = tf.nn.rnn_cell.GRUCell(rnn_units, name="{}_backward_RNN".format(self.scope))
    
    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("conv_bank"):
                
                conv_outputs = tf.concat(
                    [conv1d(inputs, k, self.conv_channels, tf.nn.relu, self.is_training, 0.,
                            "conv1d_{}".format(k)) for k in range(1, self.K + 1)],
                    axis=-1
                )
            
            
            maxpool_output = tf.layers.max_pooling1d(
                conv_outputs,
                pool_size=self.pool_size,
                strides=1,
                padding="same")
            
            
            proj1_output = conv1d(maxpool_output, self.projection_kernel_size, self.projections[0],
                                  tf.nn.relu, self.is_training, 0., "proj1")
            proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections[1],
                                  lambda _: _, self.is_training, 0., "proj2")
            
            
            highway_input = proj2_output + inputs
            
            
            if highway_input.shape[2] != self.highway_units:
                highway_input = tf.layers.dense(highway_input, self.highway_units)
            
            
            for highwaynet in self.highwaynet_layers:
                highway_input = highwaynet(highway_input)
            rnn_input = highway_input
            
            
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell,
                self._bw_cell,
                rnn_input,
                sequence_length=input_lengths,
                dtype=tf.float32)
            return tf.concat(outputs, axis=2)  


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    
    
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0.,
                 state_is_tuple=True, name=None):
        
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)
        
        if zm < 0. or zs > 1.:
            raise ValueError("One/both provided Zoneout factors are not in [0, 1]")
        
        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self, inputs, state, scope=None):
        
        output, new_state = self._cell(inputs, state, scope)
        
        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else \
				self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])
        
        
        if self.is_training:
            
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c,
                                                         (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h,
                                                            (1 - self._zoneout_outputs)) + prev_h
        
        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h
        
        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c,
                                                                                                  h])
        
        return output, new_state


class EncoderConvolutions:
    
    
    def __init__(self, is_training, hparams, activation=tf.nn.relu, scope=None):
        
        super(EncoderConvolutions, self).__init__()
        self.is_training = is_training
        
        self.kernel_size = hparams.enc_conv_kernel_size
        self.channels = hparams.enc_conv_channels
        self.activation = activation
        self.scope = "enc_conv_layers" if scope is None else scope
        self.drop_rate = hparams.tacotron_dropout_rate
        self.enc_conv_num_layers = hparams.enc_conv_num_layers
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.enc_conv_num_layers):
                x = conv1d(x, self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate,
                           "conv_layer_{}_".format(i + 1) + self.scope)
        return x


class EncoderRNN:
    
    
    def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
        
        super(EncoderRNN, self).__init__()
        self.is_training = is_training
        
        self.size = size
        self.zoneout = zoneout
        self.scope = "encoder_LSTM" if scope is None else scope
        
        
        self._fw_cell = ZoneoutLSTMCell(size, is_training,
                                        zoneout_factor_cell=zoneout,
                                        zoneout_factor_output=zoneout,
                                        name="encoder_fw_LSTM")
        
        
        self._bw_cell = ZoneoutLSTMCell(size, is_training,
                                        zoneout_factor_cell=zoneout,
                                        zoneout_factor_output=zoneout,
                                        name="encoder_bw_LSTM")
    
    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell,
                self._bw_cell,
                inputs,
                sequence_length=input_lengths,
                dtype=tf.float32,
                swap_memory=True)
            
            return tf.concat(outputs, axis=2)  


class Prenet:
    
    
    def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu,
                 scope=None):
        
        super(Prenet, self).__init__()
        self.drop_rate = drop_rate
        
        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training
        
        self.scope = "prenet" if scope is None else scope
    
    def __call__(self, inputs):
        x = inputs
        
        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                dense = tf.layers.dense(x, units=size, activation=self.activation,
                                        name="dense_{}".format(i + 1))
                
                x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
                                      name="dropout_{}".format(i + 1) + self.scope)
        return x


class DecoderRNN:
    
    
    def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
        
        super(DecoderRNN, self).__init__()
        self.is_training = is_training
        
        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.scope = "decoder_rnn" if scope is None else scope
        
        
        self.rnn_layers = [ZoneoutLSTMCell(size, is_training,
                                           zoneout_factor_cell=zoneout,
                                           zoneout_factor_output=zoneout,
                                           name="decoder_LSTM_{}".format(i + 1)) for i in
                           range(layers)]
        
        self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)
    
    def __call__(self, inputs, states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs, states)


class FrameProjection:
    
    
    def __init__(self, shape=80, activation=None, scope=None):
        
        super(FrameProjection, self).__init__()
        
        self.shape = shape
        self.activation = activation
        
        self.scope = "Linear_projection" if scope is None else scope
        self.dense = tf.layers.Dense(units=shape, activation=activation,
                                     name="projection_{}".format(self.scope))
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            
            output = self.dense(inputs)
            
            return output


class StopProjection:
    
    
    def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
        
        super(StopProjection, self).__init__()
        self.is_training = is_training
        
        self.shape = shape
        self.activation = activation
        self.scope = "stop_token_projection" if scope is None else scope
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     activation=None, name="projection_{}".format(self.scope))
            
            
            if self.is_training:
                return output
            return self.activation(output)


class Postnet:
    
    
    def __init__(self, is_training, hparams, activation=tf.nn.tanh, scope=None):
        
        super(Postnet, self).__init__()
        self.is_training = is_training
        
        self.kernel_size = hparams.postnet_kernel_size
        self.channels = hparams.postnet_channels
        self.activation = activation
        self.scope = "postnet_convolutions" if scope is None else scope
        self.postnet_num_layers = hparams.postnet_num_layers
        self.drop_rate = hparams.tacotron_dropout_rate
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.postnet_num_layers - 1):
                x = conv1d(x, self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate,
                           "conv_layer_{}_".format(i + 1) + self.scope)
            x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training,
                       self.drop_rate,
                       "conv_layer_{}_".format(5) + self.scope)
        return x


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=None,
            padding="same")
        batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
        activated = activation(batched)
        return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
                                 name="dropout_{}".format(scope))


def _round_up_tf(x, multiple):
    
    remainder = tf.mod(x, multiple)
    
    x_round = tf.cond(tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
                      lambda: x,
                      lambda: x + multiple - remainder)
    
    return x_round


def sequence_mask(lengths, r, expand=True):
    
    max_len = tf.reduce_max(lengths)
    max_len = _round_up_tf(max_len, tf.convert_to_tensor(r))
    if expand:
        return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
    return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)


def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
    
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)
    
    
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]],
                   dtype=tf.float32)
    mask_ = mask * ones
    
    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
        return tf.losses.mean_squared_error(labels=targets, predictions=outputs, weights=mask_)


def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
    
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, False)
    
    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
        
        losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs,
                                                          pos_weight=hparams.cross_entropy_pos_weight)
    
    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
        masked_loss = losses * mask
    
    return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)


def MaskedLinearLoss(targets, outputs, targets_lengths, hparams, mask=None):
    
    if mask is None:
        mask = sequence_mask(targets_lengths, hparams.outputs_per_step, True)
    
    
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]],
                   dtype=tf.float32)
    mask_ = mask * ones
    
    l1 = tf.abs(targets - outputs)
    n_priority_freq = int(2000 / (hparams.sample_rate * 0.5) * hparams.num_freq)
    
    with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask_))]):
        masked_l1 = l1 * mask_
        masked_l1_low = masked_l1[:, :, 0:n_priority_freq]
    
    mean_l1 = tf.reduce_sum(masked_l1) / tf.reduce_sum(mask_)
    mean_l1_low = tf.reduce_sum(masked_l1_low) / tf.reduce_sum(mask_)
    
    return 0.5 * mean_l1 + 0.5 * mean_l1_low
