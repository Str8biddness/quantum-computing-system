"""Quantum Circuit Simulator - Circuit Representation Module

This module provides the core quantum circuit representation and manipulation
functionality for the quantum computing system.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QuantumCircuit:
    """Represents a quantum circuit with multi-qubit states and gate operations."""
    
    def __init__(self, num_qubits: int, name: str = "quantum_circuit"):
        """Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            name: Optional name for the circuit
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        if num_qubits > 32:
            raise ValueError("Maximum 32 qubits supported")
            
        self.num_qubits = num_qubits
        self.name = name
        self.gates = []
        self.measurements = []
        self._state_vector = None
        self._initialize_state()
        
        logger.info(f"Created quantum circuit '{name}' with {num_qubits} qubits")
    
    def _initialize_state(self):
        """Initialize circuit to |0...0> state."""
        from .state import QuantumState
        self._state_vector = QuantumState(self.num_qubits)
    
    def add_gate(self, gate, *qubit_indices):
        """Add a quantum gate to the circuit.
        
        Args:
            gate: Gate object to apply
            *qubit_indices: Target qubit indices
        """
        # Validate qubit indices
        for idx in qubit_indices:
            if idx < 0 or idx >= self.num_qubits:
                raise ValueError(f"Invalid qubit index {idx}")
        
        self.gates.append((gate, qubit_indices))
        logger.debug(f"Added gate {gate.__class__.__name__} to qubits {qubit_indices}")
        return self
    
    def apply_gates(self):
        """Apply all gates in the circuit to the state vector."""
        for gate, qubit_indices in self.gates:
            self._state_vector.apply_gate(gate, qubit_indices)
        logger.info(f"Applied {len(self.gates)} gates to circuit")
    
    def measure(self, qubit_index: int, shots: int = 1) -> Dict[str, int]:
        """Measure a specific qubit.
        
        Args:
            qubit_index: Index of qubit to measure
            shots: Number of measurement shots
            
        Returns:
            Dictionary of measurement results and counts
        """
        from .measurement import measure_qubit
        results = measure_qubit(self._state_vector, qubit_index, shots)
        self.measurements.append((qubit_index, results))
        return results
    
    def measure_all(self, shots: int = 1024) -> Dict[str, int]:
        """Measure all qubits in the circuit.
        
        Args:
            shots: Number of measurement shots
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        from .measurement import measure_all_qubits
        self.apply_gates()
        results = measure_all_qubits(self._state_vector, shots)
        logger.info(f"Measured all qubits with {shots} shots")
        return results
    
    def get_statevector(self) -> np.ndarray:
        """Get the current state vector.
        
        Returns:
            State vector as numpy array
        """
        return self._state_vector.get_vector()
    
    def reset(self):
        """Reset circuit to initial |0...0> state."""
        self._initialize_state()
        self.measurements = []
        logger.debug(f"Reset circuit '{self.name}'")
    
    def depth(self) -> int:
        """Calculate circuit depth (number of gate layers).
        
        Returns:
            Circuit depth
        """
        # Simplified depth calculation
        return len(self.gates)
    
    def width(self) -> int:
        """Get circuit width (number of qubits).
        
        Returns:
            Number of qubits
        """
        return self.num_qubits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit to dictionary representation.
        
        Returns:
            Dictionary with circuit information
        """
        return {
            'name': self.name,
            'num_qubits': self.num_qubits,
            'num_gates': len(self.gates),
            'depth': self.depth(),
            'measurements': len(self.measurements)
        }
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(name='{self.name}', qubits={self.num_qubits}, gates={len(self.gates)})"
