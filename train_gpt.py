import numpy as np
import math


def softmax(x, axis=1):
    """Apply softmax to an input tensor.
    Args:
        x(np.array): input tensor
        axis(int): axis to apply softmax
    Returns:
        np.array: output tensor after applying softmax
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def attention_forward(out, preatt, att, inp, B, T, C, NH):
    """
    Forward method for the attention layer.
    Args:
        out(np.array): output tensor of shape (B, T, C)
        preatt(np.array): pre-attention tensor of shape (B, NH, T, T)
        att(np.array): attention tensor of shape (B, NH, T, T)
        inp(np.array): input tensor of shape (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        T(int): sequence length
        C(int): number of channels(features)
        NH(int): number of heads
        B(int): batch size
    Returns:
        out(np.array): updated output Tensor
    """
    hs = C // NH
    for b in range(B):
        for h in range(NH):
            # Separate query, key, value tensors
            query = inp[b, :, h * hs : (h + 1) * hs]
            key = inp[b, :, C + h * hs : C + (h + 1) * hs]
            value = inp[b, :, 2 * C + h * hs : 2 * C + (h + 1) * hs]

            # Q*K.T/(sqrt(head size))
            preatt[b, h, :, :] = np.matmul(query, key.T) / math.sqrt(hs)

            # Apply causal masking during training
            mask = np.triu(np.ones((T, T)), k=1)  # Upper triangular matrix
            preatt[b, h, :, :] -= mask * 1e10  # Apply a large negative to future steps

            # apply softmax
            att[b, h, :, :] = softmax(preatt[b, h, :, :])

            # softmax * value
            out[b, :, h * hs : (h + 1) * hs] = np.matmul(att[b, h, :, :], value)

    return out


if __name__ == "__main__":
    B = 1
    T = 2
    C = 4
    NH = 2
    inp = np.array(
        [
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            ]
        ]
    )
    out = np.zeros((B, T, C))
    preatt = np.zeros((B, NH, T, T))
    att = np.zeros((B, NH, T, T))
    att_result = attention_forward(out, preatt, att, inp, B, T, C, NH)
    assert np.allclose(att_result, np.array([[[9, 10, 11, 12], [21, 22, 23, 24]]]))
