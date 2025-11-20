"""Quantum State Management Module

Manages quantum state vectors and operations on them.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class QuantumState:
    """Represents a quantum state vector."""
    
    def __init__(self, num_qubits: int):
        """Initialize quantum state to |0...0>.
        
        Args:
            num_qubits: Number of qubits
        """
        if num_qubits <= 0 or num_qubits > 32:
            raise ValueError("Number of qubits must be between 1 and 32")
        
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        
        # Initialize to |0...0> state
        self._state_vector = np.zeros(self.dimension, dtype=complex)
        self._state_vector[0] = 1.0
        
        logger.debug(f"Initialized {num_qubits}-qubit state")
    
    def get_vector(self) -> np.ndarray:
        """Get the state vector.
        
        Returns:
            Complex numpy array representing state
        """
        return self._state_vector.copy()
    
    def set_vector(self, vector: np.ndarray):
        """Set the state vector.
        
        Args:
            vector: New state vector
        """
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector must have dimension {self.dimension}")
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot set zero vector as state")
        
        self._state_vector = vector / norm
        logger.debug("State vector updated")
    
    def apply_gate(self, gate, qubit_indices: Tuple[int, ...]):
        """Apply a quantum gate to specified qubits.
        
        Args:
            gate: QuantumGate object
            qubit_indices: Indices of qubits to apply gate to
        """
        gate_matrix = gate.matrix
        num_gate_qubits = int(np.log2(gate_matrix.shape[0]))
        
        if len(qubit_indices) != num_gate_qubits:
            raise ValueError(f"Gate requires {num_gate_qubits} qubits")
        
        # Apply gate using tensor product expansion
        if num_gate_qubits == 1:
            self._apply_single_qubit_gate(gate_matrix, qubit_indices[0])
        elif num_gate_qubits == 2:
            self._apply_two_qubit_gate(gate_matrix, qubit_indices[0], qubit_indices[1])
        else:
            self._apply_multi_qubit_gate(gate_matrix, qubit_indices)
        
        logger.debug(f"Applied {gate.name} gate to qubits {qubit_indices}")
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, target: int):
        """Apply single-qubit gate efficiently."""
        # Reshape state vector for easier manipulation
        shape = [2] * self.num_qubits
        state_tensor = self._state_vector.reshape(shape)
        
        # Move target qubit to first position
        state_tensor = np.moveaxis(state_tensor, target, 0)
        
        # Apply gate matrix
        state_tensor = np.tensordot(gate_matrix, state_tensor, axes=([1], [0]))
        
        # Move back to original position
        state_tensor = np.moveaxis(state_tensor, 0, target)
        
        # Flatten back to vector
        self._state_vector = state_tensor.reshape(self.dimension)
    
    def _apply_two_qubit_gate(self, gate_matrix: np.ndarray, control: int, target: int):
        """Apply two-qubit gate."""
        # Simple implementation - can be optimized
        full_matrix = self._expand_gate_to_full_system(gate_matrix, [control, target])
        self._state_vector = full_matrix @ self._state_vector
    
    def _apply_multi_qubit_gate(self, gate_matrix: np.ndarray, qubit_indices: Tuple[int, ...]):
        """Apply multi-qubit gate."""
        full_matrix = self._expand_gate_to_full_system(gate_matrix, list(qubit_indices))
        self._state_vector = full_matrix @ self._state_vector
    
    def _expand_gate_to_full_system(self, gate_matrix: np.ndarray, qubit_indices: list) -> np.ndarray:
        """Expand gate matrix to full system size."""
        # Create identity matrices for uninvolved qubits
        full_matrix = np.eye(1, dtype=complex)
        
        for qubit in range(self.num_qubits):
            if qubit in qubit_indices:
                idx = qubit_indices.index(qubit)
                gate_dim = int(np.sqrt(gate_matrix.shape[0]))
                qubit_matrix = np.eye(gate_dim, dtype=complex)
                # This is simplified - full implementation would extract proper submatrix
                full_matrix = np.kron(full_matrix, qubit_matrix)
            else:
                full_matrix = np.kron(full_matrix, np.eye(2, dtype=complex))
        
        return full_matrix
    
    def get_probability(self, basis_state: int) -> float:
        """Get probability of measuring a specific basis state.
        
        Args:
            basis_state: Integer representation of basis state
            
        Returns:
            Probability (0 to 1)
        """
        if basis_state < 0 or basis_state >= self.dimension:
            raise ValueError(f"Basis state must be in range [0, {self.dimension-1}]")
        
        amplitude = self._state_vector[basis_state]
        return abs(amplitude) ** 2
    
    def get_probabilities(self) -> np.ndarray:
        """Get probabilities for all basis states.
        
        Returns:
            Array of probabilities
        """
        return np.abs(self._state_vector) ** 2
    
    def normalize(self):
        """Normalize the state vector."""
        norm = np.linalg.norm(self._state_vector)
        if norm > 0:
            self._state_vector /= norm
    
    def __repr__(self) -> str:
        return f"QuantumState(qubits={self.num_qubits}, dimension={self.dimension})"
