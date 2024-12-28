import numpy as np
import os

# Constants
TIME_FRAMES = 1000  # Number of time samples
OUTPUT_DIR = 'sample_inputs'  # Directory to save the generated files
MOD_TYPES = ['QPSK', '8PSK', '16APSK', '32APSK', '64APSK']  # Added 64APSK

# Functions to generate modulations
def generate_qpsk(n_samples):
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return np.random.choice(symbols, n_samples)

def generate_psk(n_samples, M):
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    symbols = np.exp(1j * angles)
    return np.random.choice(symbols, n_samples)

def generate_apsk(n_samples, M):
    if M == 16:
        radii = [1, 2]
        symbols = []
        for r in radii:
            angles = np.linspace(0, 2 * np.pi, M // len(radii), endpoint=False)
            symbols.extend(r * np.exp(1j * angles))
    elif M == 32:
        radii = [1, 2, 3]
        symbols = []
        for r in radii:
            angles = np.linspace(0, 2 * np.pi, M // len(radii), endpoint=False)
            symbols.extend(r * np.exp(1j * angles))
    elif M == 64:  # Added 64APSK
        radii = [1, 2, 3, 4]
        symbols = []
        for r in radii:
            angles = np.linspace(0, 2 * np.pi, M // len(radii), endpoint=False)
            symbols.extend(r * np.exp(1j * angles))
    else:
        raise ValueError("Unsupported M for APSK. Use 16, 32, or 64.")
    return np.random.choice(symbols, n_samples)

# Generate modulation samples and save to files
def generate_sample_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for mod_type in MOD_TYPES:
        if mod_type == 'QPSK':
            signal = generate_qpsk(TIME_FRAMES)
        elif mod_type == '8PSK':
            signal = generate_psk(TIME_FRAMES, 8)
        elif mod_type == '16APSK':
            signal = generate_apsk(TIME_FRAMES, 16)
        elif mod_type == '32APSK':
            signal = generate_apsk(TIME_FRAMES, 32)
        elif mod_type == '64APSK':  # Added 64APSK handling
            signal = generate_apsk(TIME_FRAMES, 64)
        else:
            raise ValueError("Unsupported modulation type.")
        
        # Save the file without a header
        file_path = os.path.join(OUTPUT_DIR, f"{mod_type}_sample.txt")
        np.savetxt(file_path, signal.view(float).reshape(-1, 2), delimiter=",")
        print(f"Sample file generated for {mod_type}: {file_path}")

if __name__ == "__main__":
    generate_sample_files()
