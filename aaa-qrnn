# Attention + Autoencoder with QRNN (TensorFlow 2.x / Keras)
# Assumes you already have X (shape (N, features)) and Y (shape (N,))
# Example usage: drop this into a script or notebook and run.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------
# Hyperparameters (change as needed)
# -----------------------
SEQ_LEN = 30           # length of each input sequence/time-window
FEATURES = 23          # input feature dimension (your X has 23)
BATCH_SIZE = 256
LATENT_DIM = 64        # latent size of encoder output (bottleneck)
QRNN_HIDDEN = 64       # hidden units in QRNN layers
KERNEL_SIZE = 2        # convolution kernel width for QRNN
EPOCHS = 30
LEARNING_RATE = 1e-3
RECON_LOSS_WEIGHT = 1.0
SUPERVISED_LOSS_WEIGHT = 1.0

# -----------------------
# Custom QRNN Layer (fo-pooling)
# -----------------------
class QRNN(layers.Layer):
    def __init__(self, units, kernel_size=2, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.conv = layers.Conv1D(filters=self.units * 3,
                                  kernel_size=self.kernel_size,
                                  padding='same',
                                  activation=None,
                                  use_bias=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, time, features)
        conv_out = self.conv(inputs)  # (batch, time, 3*units)
        z, f, o = tf.split(conv_out, 3, axis=-1)
        z = tf.tanh(z)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)

        # transpose for time-major iteration
        z_t = tf.transpose(z, perm=[1, 0, 2])  # (time, batch, units)
        f_t = tf.transpose(f, perm=[1, 0, 2])
        o_t = tf.transpose(o, perm=[1, 0, 2])

        time_steps = tf.shape(z_t)[0]
        batch_size = tf.shape(z_t)[1]

        init_c = tf.zeros((batch_size, self.units), dtype=inputs.dtype)

        # use while_loop to compute hidden states
        def body(t, c_prev, h_list):
            zcur = z_t[t]
            fcur = f_t[t]
            ocur = o_t[t]
            ccur = fcur * c_prev + (1.0 - fcur) * zcur
            hcur = ocur * ccur
            h_list = h_list.write(t, hcur)
            return t + 1, ccur, h_list

        def cond(t, c_prev, h_list):
            return t < time_steps

        h_ta = tf.TensorArray(dtype=inputs.dtype, size=time_steps)
        _, _, h_ta = tf.while_loop(cond, body, loop_vars=[0, init_c, h_ta])
        h_seq = h_ta.stack()  # (time, batch, units)
        h_seq = tf.transpose(h_seq, [1, 0, 2])  # (batch, time, units)

        if self.return_sequences:
            return h_seq
        else:
            return h_seq[:, -1, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "kernel_size": self.kernel_size,
            "return_sequences": self.return_sequences
        })
        return cfg

# -----------------------
# Bahdanau Attention layer (works with encoder outputs and decoder state)
# -----------------------
class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, enc_outputs, dec_hidden):
        # enc_outputs: (batch, time_enc, enc_units)
        # dec_hidden: (batch, dec_units)  or (batch, time_dec, dec_units) for stepwise; here we assume single vector (last state)
        # expand dec_hidden to time axis
        dec_hidden_time = tf.expand_dims(dec_hidden, axis=1)  # (batch, 1, dec_units)
        score = self.V(tf.nn.tanh(self.W1(enc_outputs) + self.W2(dec_hidden_time)))  # (batch, time_enc, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, time_enc, 1)
        context_vector = tf.reduce_sum(attention_weights * enc_outputs, axis=1)  # (batch, enc_units)
        # return context and weights (weights useful for inspection)
        return context_vector, tf.squeeze(attention_weights, -1)

# -----------------------
# Build the encoder-decoder autoencoder with attention
# -----------------------
def build_model(seq_len=SEQ_LEN, features=FEATURES,
                qrnn_hidden=QRNN_HIDDEN, latent_dim=LATENT_DIM,
                kernel_size=KERNEL_SIZE):
    # Encoder input: sequences
    enc_inputs = layers.Input(shape=(seq_len, features), name='encoder_input')

    # Encoder QRNN stack (return sequences for attention)
    x = QRNN(qrnn_hidden, kernel_size=kernel_size, return_sequences=True, name='qrnn_enc_1')(enc_inputs)
    x = layers.Dropout(0.1)(x)
    x = QRNN(qrnn_hidden, kernel_size=kernel_size, return_sequences=True, name='qrnn_enc_2')(x)
    enc_outputs = layers.LayerNormalization()(x)  # (batch, time, qrnn_hidden)

    # Bottleneck: compress last time hidden to latent representation
    enc_last = layers.Lambda(lambda t: t[:, -1, :])(enc_outputs)  # (batch, hidden)
    latent = layers.Dense(latent_dim, activation='relu', name='latent_dense')(enc_last)

    # Decoder: optional initial input -- we will use zeros as decoder input and rely on attention + latent projection
    # Prepare repeated latent to feed decoder per time-step
    dec_initial = layers.RepeatVector(seq_len)(latent)  # (batch, seq_len, latent_dim)
    # project latent to decoder input dim
    dec_in = layers.TimeDistributed(layers.Dense(qrnn_hidden))(dec_initial)

    # Decoder QRNN (return sequences)
    dec_x = QRNN(qrnn_hidden, kernel_size=kernel_size, return_sequences=True, name='qrnn_dec_1')(dec_in)

    # Attention at each decoder time-step: here we compute attention using encoder outputs and decoder last hidden.
    # For simplicity we compute attention using decoder's last hidden state (global-context attention)
    dec_last = layers.Lambda(lambda t: t[:, -1, :])(dec_x)  # (batch, hidden)

    attention_layer = BahdanauAttention(units=qrnn_hidden)
    context_vector, attn_weights = attention_layer(enc_outputs, dec_last)  # context_vector: (batch, enc_units)

    # Concat context to every decoder time-step output (broadcast)
    context_repeated = layers.RepeatVector(seq_len)(context_vector)  # (batch, seq_len, enc_units)
    dec_combined = layers.Concatenate(axis=-1)([dec_x, context_repeated])

    # Final decoder projection to reconstruct input features (per timestep)
    recon_seq = layers.TimeDistributed(layers.Dense(features), name='reconstruction')(dec_combined)  # (batch, seq_len, features)

    # Supervised head: predict Y (we use latent vector -> fully connected)
    y_pred = layers.Dense(1, activation=None, name='y_pred')(latent)

    model = models.Model(inputs=enc_inputs, outputs=[recon_seq, y_pred])
    return model

# -----------------------
# Data prep: sliding windows, scaling
# -----------------------
def create_windows(X, Y, seq_len=SEQ_LEN):
    """
    Convert raw X (N, features) and Y (N,) to supervised windows.
    For window covering indices [i, i+seq_len-1], the target y is Y[i+seq_len-1] (last step).
    Returns X_windows shape (M, seq_len, features), Y_windows shape (M,)
    """
    N = X.shape[0]
    M = N - seq_len + 1
    Xw = np.zeros((M, seq_len, X.shape[1]), dtype=X.dtype)
    Yw = np.zeros((M,), dtype=Y.dtype)
    for i in range(M):
        Xw[i] = X[i:i+seq_len]
        Yw[i] = Y[i+seq_len-1]
    return Xw, Yw

# -----------------------
# Example usage with your X and Y arrays
# -----------------------
def train_example(X_raw, Y_raw):
    # 1) Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)  # scale features columnwise

    # 2) windows
    Xw, Yw = create_windows(X_scaled, Y_raw, seq_len=SEQ_LEN)
    print("Windows produced:", Xw.shape, Yw.shape)

    # 3) train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(Xw, Yw, test_size=0.15, shuffle=True, random_state=42)

    # 4) build model
    model = build_model(seq_len=SEQ_LEN, features=X_train.shape[-1],
                        qrnn_hidden=QRNN_HIDDEN, latent_dim=LATENT_DIM,
                        kernel_size=KERNEL_SIZE)
    model.summary()

    # 5) compile with two losses: reconstruction and supervised
    loss_weights = {'reconstruction': RECON_LOSS_WEIGHT, 'y_pred': SUPERVISED_LOSS_WEIGHT}
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss={'reconstruction': 'mse', 'y_pred': 'mse'},
                  loss_weights=loss_weights,
                  metrics={'reconstruction': ['mse'], 'y_pred': ['mse']})

    # 6) training
    history = model.fit(X_train, {'reconstruction': X_train, 'y_pred': Y_train},
                        validation_data=(X_val, {'reconstruction': X_val, 'y_pred': Y_val}),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=2)
    return model, history, scaler

# -----------------------
# If you run this script, feed actual X and Y arrays here
# -----------------------
if __name__ == "__main__":
    # Example: create random toy data if actual X/Y not provided
    # Replace these with your real arrays (numpy arrays)
    N = 374247  # your sample count
    # Here we create toy X/Y for demonstration; comment-out if you pass real X,Y
    # X = np.random.randn(N, FEATURES).astype(np.float32)
    # Y = np.random.randn(N).astype(np.float32)

    # If you have actual X and Y:
    try:
        X  # if variable exists in the environment
        Y
    except NameError:
        # For demonstration: generate small synthetic subset to avoid long runs
        N_demo = 500
        X = np.random.randn(N_demo, FEATURES).astype(np.float32)
        Y = (np.sum(X[:, :3], axis=1) + np.random.randn(N_demo)*0.1).astype(np.float32)

    model, history, scaler = train_example(X, Y)
