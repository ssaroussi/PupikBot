import numpy as np


H = 128 # Number of LSTM layer's neurons
D =  4015463 # Number of input dimension == number of items in vocabulary
Z = H + D # Because we will concatenate LSTM state with the input

model = dict(
    Wf=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wi=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wc=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wo=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
    bf=np.zeros((1, H)),
    bi=np.zeros((1, H)),
    bc=np.zeros((1, H)),
    bo=np.zeros((1, H)),
    by=np.zeros((1, D))
)

def lstm_forward(X, state):
    m = model
    Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
    bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

    h_old, c_old = state

    # One-hot encode
    X_one_hot = np.zeros(D)
    X_one_hot[X] = 1.
    X_one_hot = X_one_hot.reshape(1, -1)

    # Concatenate old state with current input
    X = np.column_stack((h_old, X_one_hot))

    hf = sigmoid(X @ Wf + bf)
    hi = sigmoid(X @ Wi + bi)
    ho = sigmoid(X @ Wo + bo)
    hc = tanh(X @ Wc + bc)

    c = hf * c_old + hi * hc
    h = ho * tanh(c)

    y = h @ Wy + by
    prob = softmax(y)

    state = (h, c) # Cache the states of current h & c for next iter
    cache = ... # Add all intermediate variables to this cache

    return prob, state, cache


def lstm_backward(prob, y_train, d_next, cache):
    # Unpack the cache variable to get the intermediate variables used in forward step
    ... = cache
    dh_next, dc_next = d_next

    # Softmax loss gradient
    dy = prob.copy()
    dy[1, y_train] -= 1.

    # Hidden to output gradient
    dWy = h.T @ dy
    dby = dy
    # Note we're adding dh_next here
    dh = dy @ Wy.T + dh_next

    # Gradient for ho in h = ho * tanh(c)
    dho = tanh(c) * dh
    dho = dsigmoid(ho) * dho

    # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
    dc = ho * dh * dtanh(c)
    dc = dc + dc_next

    # Gradient for hf in c = hf * c_old + hi * hc
    dhf = c_old * dc
    dhf = dsigmoid(hf) * dhf

    # Gradient for hi in c = hf * c_old + hi * hc
    dhi = hc * dc
    dhi = dsigmoid(hi) * dhi

    # Gradient for hc in c = hf * c_old + hi * hc
    dhc = hi * dc
    dhc = dtanh(hc) * dhc

    # Gate gradients, just a normal fully connected layer gradient
    dWf = X.T @ dhf
    dbf = dhf
    dXf = dhf @ Wf.T

    dWi = X.T @ dhi
    dbi = dhi
    dXi = dhi @ Wi.T

    dWo = X.T @ dho
    dbo = dho
    dXo = dho @ Wo.T

    dWc = X.T @ dhc
    dbc = dhc
    dXc = dhc @ Wc.T

    # As X was used in multiple gates, the gradient must be accumulated here
    dX = dXo + dXc + dXi + dXf
    # Split the concatenated X, so that we get our gradient of h_old
    dh_next = dX[:, :H]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf * dc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    state = (dh_next, dc_next)

    return grad, state


def train_step(X_train, y_train, state):
    probs = []
    caches = []
    loss = 0.
    h, c = state

    # Forward Step

    for x, y_true in zip(X_train, y_train):
        prob, state, cache = lstm_forward(x, state, train=True)
        loss += cross_entropy(prob, y_true)

        # Store forward step result to be used in backward step
        probs.append(prob)
        caches.append(cache)

    # The loss is the average cross entropy
    loss /= X_train.shape[0]

    # Backward Step

    # Gradient for dh_next and dc_next is zero for the last timestep
    d_next = (np.zeros_like(h), np.zeros_like(c))
    grads = {k: np.zeros_like(v) for k, v in model.items()}

    # Go backward from the last timestep to the first
    for prob, y_true, cache in reversed(list(zip(probs, y_train, caches))):
        grad, d_next = lstm_backward(prob, y_true, d_next, cache)

        # Accumulate gradients from all timesteps
        for k in grads.keys():
            grads[k] += grad[k]

    return grads, loss, state
