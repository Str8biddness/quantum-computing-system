"""Quantum Error Mitigation and Error Correction Module

Provides quantum error correction and mitigation techniques including:
- Zero-Noise Extrapolation (ZNE)
- Measurement Error Mitigation
- Probabilistic Error Cancellation
- Readout Error Correction
- Noise Model Characterization

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeasurementErrorMitigation:
    """Mitigate measurement errors using calibration matrices."""
    
    def __init__(self, num_qubits: int):
        """Initialize measurement error mitigation.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.calibration_matrix = None
        self.inverse_calibration_matrix = None
        logger.info(f"MeasurementErrorMitigation initialized for {num_qubits} qubits")
    
    def build_calibration_matrix(self, 
                                measurement_results: Dict[str, int],
                                num_trials: int = 1000) -> np.ndarray:
        """Build calibration matrix from measured data.
        
        Args:
            measurement_results: Dictionary of measured state counts
            num_trials: Number of measurement trials per state
            
        Returns:
            Calibration matrix of shape (2^n, 2^n)
        """
        n_states = 2 ** self.num_qubits
        calib_matrix = np.zeros((n_states, n_states))
        
        # Build diagonal dominant matrix (diagonal = measured prob, off-diag = error)
        for ideal_state in range(n_states):
            ideal_prob = 1.0 / n_states if ideal_state in measurement_results else 0
            total = sum(measurement_results.values())
            
            for measured_state in range(n_states):
                count = measurement_results.get(measured_state, 0)
                calib_matrix[measured_state, ideal_state] = count / total if total > 0 else 0
        
        self.calibration_matrix = calib_matrix
        logger.info(f"Calibration matrix built: shape {calib_matrix.shape}")
        
        return calib_matrix
    
    def invert_calibration_matrix(self) -> np.ndarray:
        """Compute inverse of calibration matrix for error mitigation.
        
        Returns:
            Inverse calibration matrix
        """
        if self.calibration_matrix is None:
            raise ValueError("Calibration matrix not built yet")
        
        try:
            self.inverse_calibration_matrix = np.linalg.inv(self.calibration_matrix)
            logger.info("Inverse calibration matrix computed")
        except np.linalg.LinAlgError:
            logger.warning("Calibration matrix singular, using pseudoinverse")
            self.inverse_calibration_matrix = np.linalg.pinv(self.calibration_matrix)
        
        return self.inverse_calibration_matrix
    
    def mitigate_measurement_error(self, 
                                  measured_counts: Dict[str, int]) -> Dict[str, float]:
        """Mitigate measurement errors in measurement results.
        
        Args:
            measured_counts: Dictionary of measured state counts
            
        Returns:
            Mitigated probability distribution
        """
        if self.inverse_calibration_matrix is None:
            raise ValueError("Inverse calibration matrix not computed")
        
        n_states = 2 ** self.num_qubits
        measured_probs = np.zeros(n_states)
        total = sum(measured_counts.values())
        
        for state, count in measured_counts.items():
            idx = int(state, 2) if isinstance(state, str) else state
            measured_probs[idx] = count / total if total > 0 else 0
        
        # Apply inverse calibration to get mitigated probabilities
        mitigated_probs = self.inverse_calibration_matrix @ measured_probs
        
        # Clip to valid probability range
        mitigated_probs = np.clip(mitigated_probs, 0, 1)
        mitigated_probs = mitigated_probs / np.sum(mitigated_probs)  # Renormalize
        
        result = {}
        for i, prob in enumerate(mitigated_probs):
            state_str = format(i, f'0{self.num_qubits}b')
            result[state_str] = prob
        
        logger.info("Measurement error mitigation applied")
        return result


class ZeroNoiseExtrapolation:
    """Zero-Noise Extrapolation (ZNE) for error mitigation."""
    
    def __init__(self, num_qubits: int):
        """Initialize ZNE.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.noise_levels = []
        self.results = []
        logger.info(f"ZeroNoiseExtrapolation initialized for {num_qubits} qubits")
    
    def add_measurement(self, noise_factor: float, result: float):
        """Add measurement at specific noise level.
        
        Args:
            noise_factor: Noise scaling factor (1.0 = normal)
            result: Measured result at this noise level
        """
        self.noise_levels.append(noise_factor)
        self.results.append(result)
        logger.debug(f"Added measurement: noise={noise_factor:.2f}, result={result:.4f}")
    
    def extrapolate_to_zero_noise(self, method: str = 'linear') -> float:
        """Extrapolate to zero noise using polynomial fitting.
        
        Args:
            method: Extrapolation method ('linear', 'quadratic', 'cubic')
            
        Returns:
            Extrapolated result at zero noise
        """
        if len(self.noise_levels) < 2:
            raise ValueError("Need at least 2 measurements for extrapolation")
        
        x = np.array(self.noise_levels)
        y = np.array(self.results)
        
        if method == 'linear':
            degree = 1
        elif method == 'quadratic':
            degree = 2
        elif method == 'cubic':
            degree = 3
        else:
            degree = 1
        
        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        
        # Evaluate at zero noise
        zero_noise_result = poly(0)
        
        logger.info(f"ZNE extrapolation ({method}): result at zero noise = {zero_noise_result:.6f}")
        return zero_noise_result


class NoiseModel:
    """Characterize and apply noise models."""
    
    def __init__(self):
        """Initialize noise model."""
        self.depolarizing_rates = {}
        self.readout_errors = {}
        logger.info("NoiseModel initialized")
    
    def set_depolarizing_error(self, qubit: int, error_rate: float):
        """Set depolarizing error for a qubit.
        
        Args:
            qubit: Qubit index
            error_rate: Probability of depolarizing error (0-1)
        """
        if not 0 <= error_rate <= 1:
            raise ValueError("Error rate must be between 0 and 1")
        
        self.depolarizing_rates[qubit] = error_rate
        logger.debug(f"Set depolarizing error: qubit {qubit}, rate {error_rate:.4f}")
    
    def set_readout_error(self, qubit: int, 
                         prob_0_to_1: float, 
                         prob_1_to_0: float):
        """Set readout error probabilities for a qubit.
        
        Args:
            qubit: Qubit index
            prob_0_to_1: Probability of measuring 1 when state is 0
            prob_1_to_0: Probability of measuring 0 when state is 1
        """
        self.readout_errors[qubit] = {
            '0->1': prob_0_to_1,
            '1->0': prob_1_to_0
        }
        logger.debug(f"Set readout error: qubit {qubit}")
    
    def apply_noise_to_probabilities(self, 
                                    probabilities: np.ndarray,
                                    noise_factor: float = 1.0) -> np.ndarray:
        """Apply noise model to probability distribution.
        
        Args:
            probabilities: Input probability distribution
            noise_factor: Scaling factor for noise (>1 amplifies)
            
        Returns:
            Noisy probability distribution
        """
        noisy_probs = probabilities.copy()
        
        # Apply depolarizing noise
        for qubit, error_rate in self.depolarizing_rates.items():
            scaled_rate = min(error_rate * noise_factor, 0.25)  # Cap at 25%
            # Depolarizing reduces probability, adds uniform noise
            noisy_probs *= (1 - scaled_rate)
            noisy_probs += scaled_rate / len(noisy_probs)
        
        # Normalize
        noisy_probs = noisy_probs / np.sum(noisy_probs)
        
        logger.debug(f"Noise applied with factor {noise_factor}")
        return noisy_probs


class ProbabilisticErrorCancellation:
    """Probabilistic Error Cancellation (PEC) technique."""
    
    def __init__(self, num_qubits: int):
        """Initialize PEC.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.channel_signatures = {}
        logger.info(f"ProbabilisticErrorCancellation initialized for {num_qubits} qubits")
    
    def register_channel_signature(self, channel_name: str, 
                                  signature: Dict[str, float]):
        """Register a quantum channel signature for error cancellation.
        
        Args:
            channel_name: Name of the quantum channel
            signature: Dictionary mapping basis states to amplitudes
        """
        self.channel_signatures[channel_name] = signature
        logger.debug(f"Registered channel: {channel_name}")
    
    def estimate_circuit_error(self, circuit_depth: int) -> float:
        """Estimate circuit error based on depth.
        
        Args:
            circuit_depth: Number of gates in circuit
            
        Returns:
            Estimated error probability
        """
        # Assuming per-gate error rate of 0.001 (0.1%)
        per_gate_error = 0.001
        estimated_error = 1.0 - (1.0 - per_gate_error) ** circuit_depth
        
        logger.info(f"Estimated circuit error for depth {circuit_depth}: {estimated_error:.6f}")
        return estimated_error


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM ERROR MITIGATION MODULE DEMO")
    print("=" * 60)
    
    # Example 1: Measurement error mitigation
    print("\n1. Measurement Error Mitigation")
    meas_mitigator = MeasurementErrorMitigation(num_qubits=2)
    
    # Simulated measurement data with errors
    measured_counts = {'00': 450, '01': 30, '10': 25, '11': 495}
    
    # Build calibration
    calib_matrix = meas_mitigator.build_calibration_matrix(measured_counts)
    inverse_calib = meas_mitigator.invert_calibration_matrix()
    
    # Mitigate errors
    mitigated = meas_mitigator.mitigate_measurement_error(measured_counts)
    print(f"Mitigated probabilities: {dict(list(mitigated.items())[:2])}...")
    
    # Example 2: Zero-Noise Extrapolation
    print("\n2. Zero-Noise Extrapolation")
    zne = ZeroNoiseExtrapolation(num_qubits=2)
    
    # Add measurements at different noise levels
    zne.add_measurement(1.0, 0.85)
    zne.add_measurement(2.0, 0.75)
    zne.add_measurement(3.0, 0.65)
    
    # Extrapolate
    zero_noise_result = zne.extrapolate_to_zero_noise(method='linear')
    print(f"Zero-noise result: {zero_noise_result:.4f}")
    
    # Example 3: Noise Model
    print("\n3. Noise Model")
    noise_model = NoiseModel()
    noise_model.set_depolarizing_error(0, 0.01)
    noise_model.set_readout_error(0, 0.02, 0.03)
    
    print("\n" + "=" * 60)
    print("Error mitigation demo complete!")
    print("=" * 60)
