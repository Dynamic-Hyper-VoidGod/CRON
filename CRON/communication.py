import numpy as np, pywt

class Communication:
    def __init__(self, K, wavelet="haar", level=1):
        self.K = K
        self.wavelet = wavelet
        self.level = level

        # Metadata required to reconstruct the DWT coefficient structure
        self.coeff_lengths = None
        self.state_length = None
        self.selected_lengths = None

    def encode(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)

        coeffs = pywt.wavedec(state, self.wavelet, level=self.level, mode="periodization")

        self.coeff_lengths = [len(c) for c in coeffs]
        self.state_length = len(state)
        
        # Approximation coefficients first, then detail coefficients.
        packed = np.concatenate(coeffs)

        if not 1 <= self.K <= len(packed):
            raise ValueError(f"K must be between 1 and {len(packed)}")

        # Retain coefficients from every band, proportionally to band size
        lengths = [len(c) for c in coeffs]
        total = sum(lengths)

        allocations = [
            int(self.K * length/total)
            for length in lengths
        ]

        # Distribute any remaining slots without exceeding band sizes
        remaining = self.K - sum(allocations)

        for i in range(len(allocations)):
            extra = min(remaining, lengths[i] - allocations[i])
            allocations[i] += extra
            remaining -= extra
            if remaining == 0:
                break

        self.selected_lengths = allocations

        selected = [
            band[:count]
            for band, count in zip(coeffs, allocations)
        ]

        return np.concatenate(selected)

    def decode(self, payload):
        if self.coeff_lengths is None or self.selected_lengths is None:
            raise RuntimeError("Call encode() before decode()"
                               "to establish the coefficient layout.")

        payload = np.asarray(payload, dtype=float).reshape(-1)

        if(len(payload)) != sum(self.selected_lengths):
            raise ValueError("Payload length does not mathc the encoded layout.")

        # Recreate each complete coefficient band with zeros
        coeffs = [
            np.zeros(length, dtype=float)
            for length in self.coeff_lengths
        ]

        # Place each transmitted segment into its original band.
        offset = 0

        for i, count in enumerate(self.selected_lengths):
            coeffs[i][:count] = payload[offset:offset+count]
            offset += count

        # Reconstruct the approximate original state using IDWT
        reconstructed = pywt.waverec(coeffs, self.wavelet, mode="periodization")
        
        return reconstructed[:self.state_length]