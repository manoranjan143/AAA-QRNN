import numpy as np
from qrnn_autoencoder import build_model

def test_model_build():
    seq_len, features = 30, 23
    model = build_model(seq_len=seq_len, features=features)
    assert model is not None
    assert len(model.inputs) > 0
    assert len(model.outputs) == 2
