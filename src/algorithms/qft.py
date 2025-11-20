"""Quantum Fourier Transform (QFT) Implementation.

The Quantum Fourier Transform is a quantum analogue of the discrete Fourier
transform and is a key component in many quantum algorithms including Shor's
factoring algorithm and quantum phase estimation.

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_simulator import QuantumCircuit

logger = logging.getLogger(__name__)


class QuantumFourierTransform:
    """Implementation of Quantum Fourier Transform.
    
    The QFT transforms the computational basis states according to:
    |j⟩ → (1/√N) Σ_k e^(2πijk/N) |k⟩
    
    where N = 2^n is the dimension of the Hilbert space.
    """
    
    def __init__(self, num_qubits: int):
        """Initialize QFT.
        
        Args:
            num_qubits: Number of qubits
        
        Raises:
            ValueError: If num_qubits < 1
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        
        logger.info(f"Initialized QFT for {num_qubits} qubits")
        logger.info(f"Hilbert space dimension: {self.dimension}")
    
    def apply_qft(self, circuit: QuantumCircuit, qubits: Optional[List[int]] = None) -> None:
        """Apply Quantum Fourier Transform to circuit.
        
        Args:
            circuit: Quantum circuit to apply QFT to
            qubits: List of qubit indices (None for all qubits)
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)
        logger.info(f"Applying QFT to {n} qubits: {qubits}")
        
        # Apply QFT algorithm
        for j in range(n):
            qubit = qubits[j]
            
            # Apply Hadamard
            circuit.apply_gate('H', qubit)
            
            # Apply controlled phase rotations
            for k in range(j + 1, n):
                control = qubits[k]
                angle = np.pi / (2 ** (k - j))
                
                # Controlled-RZ gate
                self._apply_controlled_phase(circuit, control, qubit, angle)
        
        # Swap qubits to reverse order
        for i in range(n // 2):
            qubit1 = qubits[i]
            qubit2 = qubits[n - 1 - i]
            self._swap_qubits(circuit, qubit1, qubit2)
        
        logger.info("QFT applied successfully")
    
    def apply_inverse_qft(self, circuit: QuantumCircuit, qubits: Optional[List[int]] = None) -> None:
        """Apply inverse Quantum Fourier Transform.
        
        Args:
            circuit: Quantum circuit to apply inverse QFT to
            qubits: List of qubit indices (None for all qubits)
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)
        logger.info(f"Applying inverse QFT to {n} qubits: {qubits}")
        
        # Inverse QFT is reverse of QFT with negated angles
        
        # Swap qubits first (reverse of QFT)
        for i in range(n // 2):
            qubit1 = qubits[i]
            qubit2 = qubits[n - 1 - i]
            self._swap_qubits(circuit, qubit1, qubit2)
        
        # Apply inverse rotations in reverse order
        for j in range(n - 1, -1, -1):
            qubit = qubits[j]
            
            # Apply controlled phase rotations (negated angles)
            for k in range(n - 1, j, -1):
                control = qubits[k]
                angle = -np.pi / (2 ** (k - j))
                self._apply_controlled_phase(circuit, control, qubit, angle)
            
            # Apply Hadamard
            circuit.apply_gate('H', qubit)
        
        logger.info("Inverse QFT applied successfully")
    
    def _apply_controlled_phase(self, circuit: QuantumCircuit, control: int, 
                                target: int, angle: float) -> None:
        """Apply controlled phase rotation.
        
        Args:
            circuit: Quantum circuit
            control: Control qubit
            target: Target qubit
            angle: Rotation angle
        """
        # Decompose controlled-RZ using available gates
        circuit.apply_gate('RZ', target, angle / 2)
        circuit.apply_gate('CNOT', control, target)
        circuit.apply_gate('RZ', target, -angle / 2)
        circuit.apply_gate('CNOT', control, target)
    
    def _swap_qubits(self, circuit: QuantumCircuit, qubit1: int, qubit2: int) -> None:
        """Swap two qubits using CNOT gates.
        
        Args:
            circuit: Quantum circuit
            qubit1: First qubit
            qubit2: Second qubit
        """
        circuit.apply_gate('CNOT', qubit1, qubit2)
        circuit.apply_gate('CNOT', qubit2, qubit1)
        circuit.apply_gate('CNOT', qubit1, qubit2)
    
    def encode_classical_data(self, data: int) -> QuantumCircuit:
        """Encode classical integer as quantum state.
        
        Args:
            data: Integer to encode (0 to 2^n - 1)
        
        Returns:
            Quantum circuit with encoded state
        
        Raises:
            ValueError: If data is out of range
        """
        if not 0 <= data < self.dimension:
            raise ValueError(f"Data must be between 0 and {self.dimension - 1}")
        
        logger.info(f"Encoding classical data: {data}")
        
        circuit = QuantumCircuit(self.num_qubits)
        
        # Encode data in binary
        binary = format(data, f'0{self.num_qubits}b')
        logger.info(f"Binary representation: {binary}")
        
        for i, bit in enumerate(binary):
            if bit == '1':
                circuit.apply_gate('X', i)
        
        return circuit
    
    def qft_state_preparation(self, target_state: int) -> QuantumCircuit:
        """Prepare quantum state using QFT.
        
        Args:
            target_state: Target computational basis state
        
        Returns:
            Circuit that prepares the state
        """
        logger.info(f"\nPreparing state |{target_state}⟩ using QFT")
        
        # Encode classical data
        circuit = self.encode_classical_data(target_state)
        
        # Apply QFT
        self.apply_qft(circuit)
        
        return circuit
    
    def demonstrate_qft_properties(self) -> None:
        """Demonstrate QFT properties and behavior."""
        logger.info("\n" + "=" * 60)
        logger.info("QFT PROPERTIES DEMONSTRATION")
        logger.info("=" * 60)
        
        # Test 1: QFT of |0⟩ state
        logger.info("\nTest 1: QFT of |0⟩ state")
        circuit1 = QuantumCircuit(self.num_qubits)
        self.apply_qft(circuit1)
        results1 = circuit1.measure_all(shots=1000)
        logger.info("Result: Equal superposition (all states equally probable)")
        logger.info(f"Measurement distribution: {dict(list(results1.items())[:5])}...")
        
        # Test 2: QFT of |1⟩ state
        logger.info("\nTest 2: QFT of |1⟩ state")
        circuit2 = QuantumCircuit(self.num_qubits)
        circuit2.apply_gate('X', self.num_qubits - 1)
        self.apply_qft(circuit2)
        results2 = circuit2.measure_all(shots=1000)
        logger.info(f"Measurement distribution: {dict(list(results2.items())[:5])}...")
        
        # Test 3: QFT and inverse QFT (should return to original)
        logger.info("\nTest 3: QFT followed by inverse QFT")
        test_state = 5 if self.dimension > 5 else 1
        circuit3 = self.encode_classical_data(test_state)
        self.apply_qft(circuit3)
        self.apply_inverse_qft(circuit3)
        results3 = circuit3.measure_all(shots=1000)
        
        # Find most common state
        max_state = max(results3.items(), key=lambda x: x[1])
        recovered = int(max_state[0], 2)
        success = recovered == test_state
        
        logger.info(f"Original state: |{test_state}⟩")
        logger.info(f"Recovered state: |{recovered}⟩")
        logger.info(f"Success: {'✓' if success else '✗'}")
        logger.info(f"Recovery probability: {max_state[1]/1000:.2%}")


def example_basic_qft():
    """Example: Basic QFT application."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Basic Quantum Fourier Transform")
    logger.info("=" * 60)
    
    qft = QuantumFourierTransform(num_qubits=3)
    
    # Create circuit with specific state
    circuit = qft.encode_classical_data(3)  # |011⟩
    
    logger.info("\nApplying QFT to |011⟩ state...")
    qft.apply_qft(circuit)
    
    # Measure
    results = circuit.measure_all(shots=1000)
    logger.info("\nMeasurement results:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:8]
    for state, count in sorted_results:
        logger.info(f"  |{int(state, 2)}⟩ ({state}): {count/1000:.2%}")


def example_inverse_qft():
    """Example: QFT and inverse QFT."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: QFT and Inverse QFT")
    logger.info("=" * 60)
    
    qft = QuantumFourierTransform(num_qubits=4)
    
    original_state = 7
    logger.info(f"\nOriginal state: |{original_state}⟩")
    
    # Encode, QFT, inverse QFT
    circuit = qft.encode_classical_data(original_state)
    logger.info("Applying QFT...")
    qft.apply_qft(circuit)
    logger.info("Applying inverse QFT...")
    qft.apply_inverse_qft(circuit)
    
    # Measure
    results = circuit.measure_all(shots=1000)
    max_state = max(results.items(), key=lambda x: x[1])
    recovered = int(max_state[0], 2)
    
    logger.info(f"\nRecovered state: |{recovered}⟩")
    logger.info(f"Match: {'✓' if recovered == original_state else '✗'}")
    logger.info(f"Fidelity: {max_state[1]/1000:.2%}")


def example_qft_properties():
    """Example: Demonstrate QFT properties."""
    qft = QuantumFourierTransform(num_qubits=3)
    qft.demonstrate_qft_properties()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    example_basic_qft()
    example_inverse_qft()
    example_qft_properties()
