import numpy as np
import math


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def attention_forward(out, preatt, att, inp, B, T, C, NH):
    # input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    # preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    # that holds the pre-attention and post-attention scores (used in backward)
    # output is (B, T, C)
    # attention is the only layer that mixes information across time
    # every other operation is applied at every (b,t) position independently
    # (and of course, no layer mixes information across batch)
    hs = C // NH
    for b in range(B):
        for h in range(NH):
            query = inp[b, :, h * hs : (h + 1) * hs]
            key = inp[b, :, C + h * hs : C + (h + 1) * hs]
            value = inp[b, :, 2 * C + h * hs : 2 * C + (h + 1) * hs]
            preatt[b, h, :, :] = np.matmul(query, key.T) / math.sqrt(hs)
            # Apply causal masking
            mask = np.triu(np.ones((T, T)), k=1)  # Upper triangular matrix
            preatt[b, h, :, :] -= mask * 1e10  # Apply a large negative to future steps

            att[b, h, :, :] = softmax(preatt[b, h, :, :])
            out[b, :, h * hs : (h + 1) * hs] = np.matmul(att[b, h, :, :], value)

    return out


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
print("attention result = ", att_result)
