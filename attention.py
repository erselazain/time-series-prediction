import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

def dot_product(x, kernel):
    return tf.squeeze(tf.matmul(x, tf.expand_dims(kernel, axis=-1)), axis=-1)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)
        if self.bias:
            eij += self.b
        eij = tf.nn.tanh(eij)

        a = tf.exp(eij)
        if mask is not None:
            a *= tf.cast(mask, tf.float32)
        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
        weighted_input = x * tf.expand_dims(a, axis=-1)
        result = tf.reduce_sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]
