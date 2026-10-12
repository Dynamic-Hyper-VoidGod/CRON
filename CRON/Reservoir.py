import numpy as np

# Creating a Reservoir class, so each reservoir can be an instance of this class
class Reservoir:
    def __init__(self, N, alpha, washout, ridge, spectral_radius, scale, ip_node, op_node):
        self.N = N
        self.alpha = alpha
        self.ws = washout
        self.rg = ridge
        self.sr = spectral_radius
        self.ie = scale
        self.ip = ip_node
        self.op = op_node

    def _init_esn(self,seed=42):
        rng = np.random.default_rng(seed=seed)
        # Initialising the weights
        Win = rng.uniform(low=-1.0, high = 1.0, size=(self.N,self.ip))
        mask = rng.uniform(low=0.00, high = 1.00, size=(self.N, self.N)) < 0.10
        W = rng.uniform(low=-1.0, high=1.0, size=(self.N, self.N))
        Wout = np.zeros((self.op, self.N))

        W *= mask
        Win *= self.ie
        eigen_vals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigen_vals))
        s_temp = self.sr/rho
        W *= s_temp

        esn = {
        "N": self.N,
        "input_size": self.ip,
        "output_size": self.op,
        "spectral_radius": self.sr,
        "leak_rate": self.alpha,
        "ridge": self.rg,
        "input_scale": self.ie,
        "Win": Win,
        "W": W,
        "Wout": Wout,
        "EVS": eigen_vals,
        "s_temp": s_temp
        }
        
        return esn

    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def _run_res(self,data, reservoir, activation=sigmoid):
        """
        This function runs the reservoir update equations.
        """
        R = reservoir
        
        state = np.zeros((R["N"], 1)) # x(t)

        states = [] # Stores states
        
        for t, value in enumerate(data):
            
            value = np.asarray(value).reshape(-1,1)
            
            state = (1 - R["leak_rate"]) * state + R["leak_rate"] * activation(R["Win"] @ value + R["W"] @ state)

            if t >= self.ws:
             states.append(state.copy())

        cache = {
            "states": states,
            "final_state": state
        }

        return cache
    
