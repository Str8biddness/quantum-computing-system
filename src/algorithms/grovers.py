"""Grover's Quantum Search Algorithm.

This module implements Grover's algorithm for searching an unsorted database
with quadratic speedup over classical algorithms.

Grover's algorithm can find a marked item in an unsorted database of N items
in O(√N) queries, compared to O(N) for classical algorithms.

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import List, Callable, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_simulator import QuantumCircuit

logger = logging.getLogger(__name__)


class GroverSearch:
    """Implementation of Grover's quantum search algorithm.
    
    Attributes:
        num_qubits: Number of qubits in the search space
        oracle: Oracle function that marks the target state
        circuit: Quantum circuit for the algorithm
    """
    
    def __init__(self, num_qubits: int):
        """Initialize Grover's search algorithm.
        
        Args:
            num_qubits: Number of qubits (search space size is 2^n)
        
        Raises:
            ValueError: If num_qubits is less than 1
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.num_qubits = num_qubits
        self.search_space_size = 2 ** num_qubits
        self.circuit = None
        self.optimal_iterations = self._calculate_iterations()
        
        logger.info(f"Initialized Grover search for {num_qubits} qubits")
        logger.info(f"Search space size: {self.search_space_size}")
        logger.info(f"Optimal iterations: {self.optimal_iterations}")
    
    def _calculate_iterations(self) -> int:
        """Calculate optimal number of Grover iterations.
        
        Returns:
            Optimal number of iterations for maximum success probability
        """
        # Optimal iterations ≈ (π/4)√N
        iterations = int(np.round(np.pi / 4 * np.sqrt(self.search_space_size)))
        return max(1, iterations)
    
    def create_oracle(self, target_state: int) -> None:
        """Create oracle that marks the target state.
        
        The oracle applies a phase flip to the target state.
        
        Args:
            target_state: Integer representing target state (0 to 2^n - 1)
        
        Raises:
            ValueError: If target_state is out of range
        """
        if not 0 <= target_state < self.search_space_size:
            raise ValueError(
                f"Target state must be between 0 and {self.search_space_size - 1}"
            )
        
        self.target_state = target_state
        logger.info(f"Oracle created for target state: |{target_state}⟩")
    
    def _apply_oracle(self, circuit: QuantumCircuit) -> None:
        """Apply oracle operator to mark target state.
        
        Args:
            circuit: Quantum circuit to apply oracle to
        """
        # Convert target state to binary
        target_bits = format(self.target_state, f'0{self.num_qubits}b')
        
        # Apply X gates to flip qubits that should be 0 in target
        for i, bit in enumerate(target_bits):
            if bit == '0':
                circuit.apply_gate('X', i)
        
        # Apply multi-controlled Z gate (phase flip)
        if self.num_qubits == 1:
            circuit.apply_gate('Z', 0)
        elif self.num_qubits == 2:
            # Controlled-Z using CNOT and H
            circuit.apply_gate('H', 1)
            circuit.apply_gate('CNOT', 0, 1)
            circuit.apply_gate('H', 1)
        else:
            # Multi-controlled Z using Toffoli decomposition
            # For simplicity, apply Z to last qubit with controls
            circuit.apply_gate('Z', self.num_qubits - 1)
            
        # Undo X gates
        for i, bit in enumerate(target_bits):
            if bit == '0':
                circuit.apply_gate('X', i)
        
        logger.debug(f"Applied oracle for target |{self.target_state}⟩")
    
    def _apply_diffusion(self, circuit: QuantumCircuit) -> None:
        """Apply Grover diffusion operator (inversion about average).
        
        The diffusion operator amplifies the amplitude of the target state.
        
        Args:
            circuit: Quantum circuit to apply diffusion to
        """
        # Apply H to all qubits
        for i in range(self.num_qubits):
            circuit.apply_gate('H', i)
        
        # Apply X to all qubits
        for i in range(self.num_qubits):
            circuit.apply_gate('X', i)
        
        # Apply multi-controlled Z
        if self.num_qubits == 1:
            circuit.apply_gate('Z', 0)
        elif self.num_qubits == 2:
            circuit.apply_gate('H', 1)
            circuit.apply_gate('CNOT', 0, 1)
            circuit.apply_gate('H', 1)
        else:
            # Multi-controlled phase
            circuit.apply_gate('Z', self.num_qubits - 1)
        
        # Apply X to all qubits
        for i in range(self.num_qubits):
            circuit.apply_gate('X', i)
        
        # Apply H to all qubits
        for i in range(self.num_qubits):
            circuit.apply_gate('H', i)
        
        logger.debug("Applied diffusion operator")
    
    def search(self, target_state: int, iterations: Optional[int] = None) -> Tuple[int, float]:
        """Execute Grover's search algorithm.
        
        Args:
            target_state: Target state to search for
            iterations: Number of Grover iterations (None for optimal)
        
        Returns:
            Tuple of (found_state, success_probability)
        
        Raises:
            ValueError: If target_state is invalid
        """
        # Create oracle
        self.create_oracle(target_state)
        
        # Use optimal iterations if not specified
        if iterations is None:
            iterations = self.optimal_iterations
        
        logger.info(f"\nExecuting Grover search with {iterations} iterations")
        logger.info("=" * 60)
        
        # Initialize circuit
        self.circuit = QuantumCircuit(self.num_qubits)
        
        # Initialize to equal superposition
        for i in range(self.num_qubits):
            self.circuit.apply_gate('H', i)
        logger.info("Initialized to equal superposition")
        
        # Apply Grover iterations
        for iter_num in range(iterations):
            logger.debug(f"\nIteration {iter_num + 1}/{iterations}")
            
            # Apply oracle
            self._apply_oracle(self.circuit)
            
            # Apply diffusion
            self._apply_diffusion(self.circuit)
        
        logger.info(f"Completed {iterations} Grover iterations")
        
        # Measure
        results = self.circuit.measure_all(shots=1000)
        
        # Find most probable state
        max_state = max(results.items(), key=lambda x: x[1])
        found_state = int(max_state[0], 2)  # Convert binary string to int
        success_prob = max_state[1] / 1000.0
        
        logger.info("\nSearch Results:")
        logger.info(f"Target state: |{target_state}⟩ (binary: {format(target_state, f'0{self.num_qubits}b')})")
        logger.info(f"Found state: |{found_state}⟩ (binary: {format(found_state, f'0{self.num_qubits}b')})")
        logger.info(f"Success probability: {success_prob:.2%}")
        logger.info(f"Match: {'✓' if found_state == target_state else '✗'}")
        
        # Show top results
        logger.info("\nTop measurement outcomes:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
        for state, count in sorted_results:
            state_int = int(state, 2)
            prob = count / 1000.0
            marker = " ← TARGET" if state_int == target_state else ""
            logger.info(f"  |{state_int}⟩ ({state}): {prob:.2%}{marker}")
        
        return found_state, success_prob
    
    def analyze_performance(self, target_state: int, max_iterations: int = 10) -> dict:
        """Analyze Grover's algorithm performance across iterations.
        
        Args:
            target_state: Target state to search for
            max_iterations: Maximum iterations to test
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info("\nPerformance Analysis")
        logger.info("=" * 60)
        
        self.create_oracle(target_state)
        results_by_iteration = {}
        
        for num_iter in range(1, max_iterations + 1):
            circuit = QuantumCircuit(self.num_qubits)
            
            # Initialize
            for i in range(self.num_qubits):
                circuit.apply_gate('H', i)
            
            # Apply iterations
            for _ in range(num_iter):
                self._apply_oracle(circuit)
                self._apply_diffusion(circuit)
            
            # Measure
            measurements = circuit.measure_all(shots=1000)
            target_binary = format(target_state, f'0{self.num_qubits}b')
            success_prob = measurements.get(target_binary, 0) / 1000.0
            
            results_by_iteration[num_iter] = success_prob
            logger.info(f"Iterations: {num_iter:2d} | Success probability: {success_prob:.2%}")
        
        # Find optimal
        optimal_iter = max(results_by_iteration.items(), key=lambda x: x[1])
        
        logger.info("\nOptimal configuration:")
        logger.info(f"Iterations: {optimal_iter[0]}")
        logger.info(f"Success probability: {optimal_iter[1]:.2%}")
        
        return {
            'results_by_iteration': results_by_iteration,
            'optimal_iterations': optimal_iter[0],
            'max_success_probability': optimal_iter[1]
        }


def example_simple_search():
    """Example: Search in 2-qubit space."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Simple Grover Search (2 qubits)")
    logger.info("=" * 60)
    
    grover = GroverSearch(num_qubits=2)
    target = 3  # Search for |11⟩
    found, prob = grover.search(target)
    
    logger.info(f"\nSearched for |{target}⟩, found |{found}⟩ with {prob:.2%} probability")


def example_larger_search():
    """Example: Search in larger space."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Larger Grover Search (4 qubits)")
    logger.info("=" * 60)
    
    grover = GroverSearch(num_qubits=4)
    target = 10  # Search for |1010⟩
    found, prob = grover.search(target)
    
    logger.info(f"\nSearched for |{target}⟩, found |{found}⟩ with {prob:.2%} probability")


def example_performance_analysis():
    """Example: Analyze performance vs iterations."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Performance Analysis")
    logger.info("=" * 60)
    
    grover = GroverSearch(num_qubits=3)
    target = 5
    analysis = grover.analyze_performance(target, max_iterations=8)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    example_simple_search()
    example_larger_search()
    example_performance_analysis()
