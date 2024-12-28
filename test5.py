import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import skew, kurtosis
import seaborn as sns
import joblib
import os

# Constants
MOD_TYPES = ['QPSK', '8PSK', '16APSK', '32APSK']
MODEL_FILE = 'modulation_model.pkl'
TIME_FRAMES = 1000  # Number of time samples per signal

# Generate Modulation Schemes
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
    else:
        raise ValueError("Unsupported M for APSK. Use 16 or 32.")
    return np.random.choice(symbols, n_samples)

# Extract features
def extract_features(data):
    """
    Extracts real-valued features from complex time-domain signal data.
    """
    features = []
    # Real and Imaginary components
    real_part = data.real
    imag_part = data.imag

    # Amplitude and Phase
    magnitude = np.abs(data)
    phase = np.angle(data)

    # Statistical features for real and imaginary parts
    features.append(np.mean(real_part))
    features.append(np.std(real_part))
    features.append(np.mean(imag_part))
    features.append(np.std(imag_part))

    # Statistical features for magnitude and phase
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.mean(phase))
    features.append(np.std(phase))

    # Skewness and kurtosis for real and imaginary parts
    features.append(skew(real_part))
    features.append(kurtosis(real_part))
    features.append(skew(imag_part))
    features.append(kurtosis(imag_part))

    # Min and max values for real, imaginary, magnitude, and phase
    features.append(np.min(real_part))
    features.append(np.max(real_part))
    features.append(np.min(imag_part))
    features.append(np.max(imag_part))
    features.append(np.min(magnitude))
    features.append(np.max(magnitude))
    features.append(np.min(phase))
    features.append(np.max(phase))

    return features

# Generate dataset
def generate_dataset(n_samples_per_mod):
    data = []
    labels = []
    for mod_type in MOD_TYPES:
        if mod_type == 'QPSK':
            signals = [generate_qpsk(TIME_FRAMES) for _ in range(n_samples_per_mod)]
        elif mod_type == '8PSK':
            signals = [generate_psk(TIME_FRAMES, 8) for _ in range(n_samples_per_mod)]
        elif mod_type == '16APSK':
            signals = [generate_apsk(TIME_FRAMES, 16) for _ in range(n_samples_per_mod)]
        elif mod_type == '32APSK':
            signals = [generate_apsk(TIME_FRAMES, 32) for _ in range(n_samples_per_mod)]
        else:
            raise ValueError("Unsupported modulation type.")
        
        for signal in signals:
            features = extract_features(signal)  # Real-valued features
            data.append(features)
            labels.append(mod_type)
    return np.array(data), np.array(labels)

# Model training and saving
def train_model(X_train, y_train, model=None):
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model):
    joblib.dump(model, MODEL_FILE)

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

# Visualize results
def plot_constellation(data, labels, title):
    plt.figure(figsize=(8, 8))
    for mod_type in np.unique(labels):
        mask = labels == mod_type
        plt.scatter(data[mask, 0], data[mask, 1], label=mod_type, alpha=0.7, s=10)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main interactive function
def interactive_program():
    print("Welcome to Modulation Detection System")

    while True:
        print("\nSelect an option:")
        print("1. Generate dataset and train/retrain model")
        print("2. Test and visualize signal from input file")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            n_samples = int(input("Enter number of samples per modulation type (e.g., 1000): "))
            
            print(f"Generating dataset with {n_samples} samples per modulation type...")
            data, labels = generate_dataset(n_samples)
            print(f"Dataset length: {len(data)}")

            model = load_model()
            model = train_model(data, labels, model)

            print("Training complete. Saving model...")
            save_model(model)

            # Evaluate model
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred))

            # Confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=MOD_TYPES, yticklabels=MOD_TYPES)
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.show()

        elif choice == '2':
            input_file = input("Enter the path to the input text file containing I, Q values: ")
            try:
                # Read the I and Q values from the file
                iq_data = np.loadtxt(input_file, delimiter=",")
                input_data = iq_data[:, 0] + 1j * iq_data[:, 1]  # Convert to complex numbers

                if input_data.shape[0] != TIME_FRAMES:
                    print(f"The file must contain {TIME_FRAMES} timeframes of I and Q values.")
                    continue

                model = load_model()
                if model is None:
                    print("No model found. Please train a model first.")
                    continue

                features = extract_features(input_data)
                features = np.array(features).reshape(1, -1)  # Reshape for prediction

                # Predict the modulation type
                y_pred = model.predict(features)
                prediction_proba = model.predict_proba(features).max()
                print(f"Predicted Modulation Type: {y_pred[0]}")
                print(f"Prediction Confidence Score: {prediction_proba:.2f}")

                # Classification Report
                print("\nClassification Report for Detection:")
                class_report = classification_report(
                    [y_pred[0]], [y_pred[0]], labels=MOD_TYPES, target_names=MOD_TYPES, zero_division=0
                )
                print(class_report)

                # Visualization 1: Constellation Diagram
                plt.figure(figsize=(8, 8))
                plt.scatter(input_data.real, input_data.imag, alpha=0.7, s=10, label='Received Signal')
                plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
                plt.title(f"Constellation Diagram\nPredicted: {y_pred[0]} (Score: {prediction_proba:.2f})")
                plt.xlabel("In-phase (I)")
                plt.ylabel("Quadrature (Q)")
                plt.legend()
                plt.grid(True)
                plt.show()

                # Visualization 2: Scatter Plot of Signal Components Over Time
                plt.figure(figsize=(12, 6))
                plt.plot(range(TIME_FRAMES), input_data.real, label="In-phase (I)", alpha=0.8)
                plt.plot(range(TIME_FRAMES), input_data.imag, label="Quadrature (Q)", alpha=0.8)
                plt.title("Scatter Plot of Signal Components Over Time")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid(True)
                plt.show()

                # Visualization 3: Scatter Plot of I vs. Q
                plt.figure(figsize=(8, 8))
                plt.scatter(input_data.real, input_data.imag, c=np.arange(TIME_FRAMES), cmap='viridis', s=10)
                plt.colorbar(label='Sample Index')
                plt.title("Scatter Plot of I vs Q")
                plt.xlabel("In-phase (I)")
                plt.ylabel("Quadrature (Q)")
                plt.grid(True)
                plt.show()

            except Exception as e:
                print(f"Error reading input file: {e}")

        elif choice == '3':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    interactive_program()
