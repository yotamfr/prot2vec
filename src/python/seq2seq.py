from __future__ import absolute_import

from recurrentshop.recurrentshop import LSTMCell, RecurrentSequential, ExtendedRNNCell
from recurrentshop.recurrentshop import get_slices

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation, Bidirectional, Dropout
from keras.layers import add, multiply, concatenate
from keras import backend as K


class LSTMDecoderCell(ExtendedRNNCell):
    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        super(LSTMDecoderCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   use_bias=False)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer, )
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer, )

        z = add([W1(x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)
        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)
        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])


class AttentionDecoderCell(ExtendedRNNCell):
    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        self.input_ndim = 3
        super(AttentionDecoderCell, self).__init__(**kwargs)

    def build_model(self, input_shape):

        input_dim = input_shape[-1]
        output_dim = self.output_dim
        input_length = input_shape[1]
        hidden_dim = self.hidden_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W3 = Dense(1,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer)

        C = Lambda(lambda x: K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
        _xC = concatenate([x, C])
        _xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)

        alpha = W3(_xC)
        alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
        alpha = Activation('softmax')(alpha)

        _x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)), output_shape=(input_dim,))([alpha, x])

        z = add([W1(_x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)

        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z0)

        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))


        return Model([x, h_tm1, c_tm1], [y, h, c])


def AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
                     batch_size=None, input_shape=None, input_length=None,
                     input_dim=None, hidden_dim=None, depth=1,
                     bidirectional=True, unroll=False, stateful=False, dropout=0.0, ):
    '''
    This is an attention Seq2seq model based on [3].
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.
    The  math:
            Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
            so H is a sequence of vectors of length m.
            Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:
    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j))
    alpha = softmax(energy)
    Where a is a feed forward network.
    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True

    encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
        encoder.forward_layer.build(shape)
        encoder.backward_layer.build(shape)
        # patch
        encoder.layer = encoder.forward_layer

    encoded = encoder(_input)
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))

    inputs = [_input]
    decoded = decoder(encoded)
    model = Model(inputs, decoded)
    return model