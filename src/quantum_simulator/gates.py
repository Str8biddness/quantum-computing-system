"""Quantum Gates Module

Defines standard quantum gates and operations for the quantum circuit simulator.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QuantumGate:
    """Base class for quantum gates."""
    
    def __init__(self, name: str, matrix: np.ndarray):
        """Initialize a quantum gate.
        
        Args:
            name: Name of the gate
            matrix: Unitary matrix representing the gate
        """
        self.name = name
        self.matrix = matrix
        self._validate_unitary()
    
    def _validate_unitary(self):
        """Validate that the matrix is unitary."""
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Gate matrix must be square")
        
        # Check if matrix is unitary (U * U† = I)
        product = self.matrix @ self.matrix.conj().T
        identity = np.eye(self.matrix.shape[0])
        
        if not np.allclose(product, identity, atol=1e-10):
            logger.warning(f"Gate {self.name} may not be unitary")
    
    def __repr__(self) -> str:
        return f"{self.name}Gate(shape={self.matrix.shape})"


class Hadamard(QuantumGate):
    """Hadamard gate - creates superposition."""
    
    def __init__(self):
        matrix = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        super().__init__("H", matrix)


class PauliX(QuantumGate):
    """Pauli-X gate (NOT gate) - bit flip."""
    
    def __init__(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        super().__init__("X", matrix)


class PauliY(QuantumGate):
    """Pauli-Y gate - bit and phase flip."""
    
    def __init__(self):
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        super().__init__("Y", matrix)


class PauliZ(QuantumGate):
    """Pauli-Z gate - phase flip."""
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        super().__init__("Z", matrix)


class Phase(QuantumGate):
    """Phase gate (S gate) - adds phase."""
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        super().__init__("S", matrix)


class TGate(QuantumGate):
    """T gate - π/4 phase gate."""
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        super().__init__("T", matrix)


class CNOT(QuantumGate):
    """Controlled-NOT gate - 2-qubit gate."""
    
    def __init__(self):
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        super().__init__("CNOT", matrix)


class Toffoli(QuantumGate):
    """Toffoli gate (CCNOT) - 3-qubit controlled-controlled-NOT."""
    
    def __init__(self):
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        super().__init__("Toffoli", matrix)


class RX(QuantumGate):
    """Rotation around X-axis."""
    
    def __init__(self, theta: float):
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        matrix = np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ], dtype=complex)
        super().__init__("RX", matrix)


class RY(QuantumGate):
    """Rotation around Y-axis."""
    
    def __init__(self, theta: float):
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        matrix = np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=complex)
        super().__init__("RY", matrix)


class RZ(QuantumGate):
    """Rotation around Z-axis."""
    
    def __init__(self, theta: float):
        matrix = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
        super().__init__("RZ", matrix)


# Common gate instances for convenience
H = Hadamard()
X = PauliX()
Y = PauliY()
Z = PauliZ()
S = Phase()
T = TGate()
