import numpy as np

class Polynomial_Readout:
    def __init__(self, ridge=1e-6):
        self.ridge = ridge
        self.Wout = None

    def transform(self, states):
        states = np.asarray(states, dtype=float)

        if states.ndim != 2:
            raise ValueError("states must have shape (timesteps, neurons)")

        bias = np.ones((states.shape[0], 1))
        return np.concatenate([bias, states, states**2], axis=1)

    def fit(self, states, targets):
        Phi = self.transform(states)
        Y = np.asarray(targets, dtype=float).reshape(-1, 1)

        if Phi.shape[0] != Y.shape[0]:
            raise ValueError("States and targets must have equal lengths")

        penalty = self.ridge * np.eye(Phi.shape[1])
        penalty[0, 0] = 0.0

        self.Wout = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ Y)
        return self

    def predict(self, states):
        if self.Wout is None:
            raise RuntimeError("Call fit() before predict()")

        return self.transform(states) @ self.Wout


