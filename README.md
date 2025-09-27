Attention + Autoencoder with QRNN (TensorFlow 2.x / Keras)

This project implements an Autoencoder with QRNN (Quasi-Recurrent Neural Network) layers and Bahdanau Attention in TensorFlow 2.x / Keras.

The model jointly performs:

Reconstruction of input sequences (unsupervised objective).

Supervised prediction of target values Y (regression head).

It is suitable for time-series forecasting, anomaly detection, and sequence-to-label tasks.

✨ Features

Custom QRNN layer with fo-pooling implemented from scratch.

Bahdanau Attention mechanism for encoder–decoder context modeling.

Dual-objective training:

Reconstruction loss (MSE between input and output sequences).

Supervised regression loss (MSE on predicted targets).

Sliding window generator for supervised time-series input preparation.

Scalable to large datasets.

📂 Project Structure
├── model_qrnn_attention.py   # Main implementation (QRNN, Attention, Autoencoder)
├── README.md                 # Project documentation

⚙️ Requirements

Python 3.7+

TensorFlow 2.x

NumPy

scikit-learn

Install dependencies:

pip install tensorflow numpy scikit-learn

🏗️ Model Architecture

Encoder

Stacked QRNN layers (with fo-pooling).

Produces sequence embeddings.

Latent bottleneck (Dense layer).

Decoder

Repeats latent vector.

QRNN layers reconstruct time-series.

Attention integrates encoder context.

Outputs reconstructed sequence.

Supervised Head

Latent → Fully connected → Predicts target Y.

📊 Data Preparation

Input data must be:

X: shape (N, features)

Y: shape (N,)

The helper create_windows() converts raw data into supervised sliding windows:

Xw, Yw = create_windows(X, Y, seq_len=30)
# Xw -> (M, 30, features)
# Yw -> (M,)

🚀 Training Example
from model_qrnn_attention import train_example

# Example synthetic data
import numpy as np
N, FEATURES = 500, 23
X = np.random.randn(N, FEATURES).astype(np.float32)
Y = (np.sum(X[:, :3], axis=1) + np.random.randn(N)*0.1).astype(np.float32)

# Train
model, history, scaler = train_example(X, Y)

📈 Outputs

Reconstruction of input sequence (reconstruction).

Supervised prediction (y_pred).

Training history with validation metrics.

🧪 Example Training Logs
Windows produced: (471, 30, 23) (471,)
Epoch 1/30
15/15 - 2s - loss: 1.2345 - reconstruction_loss: 0.9876 - y_pred_loss: 0.2469 - val_loss: 1.2101 ...
...

🔧 Hyperparameters

Key configurable parameters:

SEQ_LEN = 30
FEATURES = 23
BATCH_SIZE = 256
LATENT_DIM = 64
QRNN_HIDDEN = 64
KERNEL_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-3

📌 Notes

Replace synthetic data with real X, Y arrays for actual tasks.

Can be adapted for classification by changing the supervised head activation and loss.

Useful for predictive maintenance, RUL estimation, or general sequence modeling.
