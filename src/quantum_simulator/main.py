"""Quantum Circuit Simulator - Main Entry Point.

This module provides example usage and demonstrations of the quantum
circuit simulator capabilities.

Examples:
    Run the script directly to see example quantum circuits:
    $ python main.py

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import sys
from typing import Dict, List
import numpy as np

# Import quantum simulator components
from quantum_simulator import (
    QuantumCircuit,
    QuantumState,
    measure_all_qubits,
    expectation_value
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_bell_state():
    """Create and measure a Bell state (maximally entangled state).
    
    Creates the state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    logger.info("=" * 60)
    logger.info("Example 1: Bell State Creation")
    logger.info("=" * 60)
    
    # Create 2-qubit circuit
    qc = QuantumCircuit(2)
    logger.info("Initial state: |00⟩")
    
    # Apply Hadamard to first qubit
    qc.apply_gate('H', 0)
    logger.info("Applied Hadamard gate to qubit 0")
    
    # Apply CNOT with control=0, target=1
    qc.apply_gate('CNOT', 0, 1)
    logger.info("Applied CNOT gate (control=0, target=1)")
    
    # Measure multiple times
    results = qc.measure_all(shots=1000)
    logger.info(f"Measurement results (1000 shots): {results}")
    
    # Verify entanglement - should see ~50% |00⟩ and ~50% |11⟩
    logger.info("Expected: ~50% |00⟩ and ~50% |11⟩ (entangled state)")
    logger.info("")


def example_superposition():
    """Demonstrate quantum superposition with multiple qubits."""
    logger.info("=" * 60)
    logger.info("Example 2: Quantum Superposition")
    logger.info("=" * 60)
    
    # Create 3-qubit circuit
    qc = QuantumCircuit(3)
    logger.info("Initial state: |000⟩")
    
    # Apply Hadamard to all qubits - creates equal superposition
    for i in range(3):
        qc.apply_gate('H', i)
    logger.info("Applied Hadamard gates to all qubits")
    logger.info("State: |+++⟩ = (|000⟩ + |001⟩ + ... + |111⟩)/√8")
    
    # Measure
    results = qc.measure_all(shots=1000)
    logger.info(f"Measurement results (1000 shots): {results}")
    logger.info("Expected: ~12.5% for each of 8 possible states")
    logger.info("")


def example_grover_search():
    """Demonstrate simplified Grover's search algorithm.
    
    Searches for marked state |11⟩ in 2-qubit space.
    """
    logger.info("=" * 60)
    logger.info("Example 3: Simplified Grover's Search")
    logger.info("=" * 60)
    
    # Create 2-qubit circuit
    qc = QuantumCircuit(2)
    logger.info("Searching for marked state |11⟩")
    
    # Initialize to equal superposition
    qc.apply_gate('H', 0)
    qc.apply_gate('H', 1)
    logger.info("Initialized to equal superposition")
    
    # Oracle: mark |11⟩ by applying phase flip
    qc.apply_gate('Z', 0)
    qc.apply_gate('Z', 1)
    logger.info("Applied oracle (phase flip to |11⟩)")
    
    # Diffusion operator (simplified)
    qc.apply_gate('H', 0)
    qc.apply_gate('H', 1)
    qc.apply_gate('X', 0)
    qc.apply_gate('X', 1)
    qc.apply_gate('CNOT', 0, 1)
    qc.apply_gate('X', 0)
    qc.apply_gate('X', 1)
    qc.apply_gate('H', 0)
    qc.apply_gate('H', 1)
    logger.info("Applied diffusion operator")
    
    # Measure
    results = qc.measure_all(shots=1000)
    logger.info(f"Measurement results (1000 shots): {results}")
    logger.info("Expected: Higher probability for |11⟩")
    logger.info("")


def example_phase_gates():
    """Demonstrate phase rotation gates."""
    logger.info("=" * 60)
    logger.info("Example 4: Phase Rotation Gates")
    logger.info("=" * 60)
    
    # Create single qubit circuit
    qc = QuantumCircuit(1)
    
    # Put qubit in superposition
    qc.apply_gate('H', 0)
    logger.info("Applied Hadamard: |+⟩ state")
    
    # Apply S gate (π/2 phase)
    qc.apply_gate('S', 0)
    logger.info("Applied S gate (π/2 phase rotation)")
    
    # Apply T gate (π/4 phase)
    qc.apply_gate('T', 0)
    logger.info("Applied T gate (π/4 phase rotation)")
    
    # Measure in X basis (apply H before measurement)
    qc.apply_gate('H', 0)
    results = qc.measure_all(shots=1000)
    logger.info(f"Measurement results (1000 shots): {results}")
    logger.info("")


def example_multi_gate_circuit():
    """Create a complex circuit with multiple gate types."""
    logger.info("=" * 60)
    logger.info("Example 5: Multi-Gate Circuit")
    logger.info("=" * 60)
    
    # Create 3-qubit circuit
    qc = QuantumCircuit(3)
    logger.info("Created 3-qubit circuit")
    
    # Apply various gates
    qc.apply_gate('H', 0)
    qc.apply_gate('X', 1)
    qc.apply_gate('Y', 2)
    logger.info("Applied: H(0), X(1), Y(2)")
    
    qc.apply_gate('CNOT', 0, 1)
    qc.apply_gate('CNOT', 1, 2)
    logger.info("Applied: CNOT(0,1), CNOT(1,2)")
    
    # Rotation gates
    qc.apply_gate('RX', 0, np.pi/4)
    qc.apply_gate('RY', 1, np.pi/3)
    qc.apply_gate('RZ', 2, np.pi/6)
    logger.info("Applied rotation gates: RX(π/4), RY(π/3), RZ(π/6)")
    
    # Get circuit info
    info = qc.get_circuit_info()
    logger.info(f"Circuit depth: {info['depth']}")
    logger.info(f"Total gates: {info['gate_count']}")
    logger.info(f"Gate types used: {info['gate_types']}")
    
    # Measure
    results = qc.measure_all(shots=1000)
    logger.info(f"Measurement results (1000 shots): {results}")
    logger.info("")


def example_expectation_value():
    """Calculate expectation values of observables."""
    logger.info("=" * 60)
    logger.info("Example 6: Expectation Value Calculation")
    logger.info("=" * 60)
    
    # Create single qubit in |+⟩ state
    qc = QuantumCircuit(1)
    qc.apply_gate('H', 0)
    logger.info("Created |+⟩ state")
    
    # Calculate expectation value of Z operator
    state = qc.get_state()
    z_observable = np.array([[1, 0], [0, -1]], dtype=complex)
    exp_val = expectation_value(state, z_observable)
    logger.info(f"⟨+|Z|+⟩ = {exp_val:.6f} (expected: 0.0)")
    
    # Create |0⟩ state
    qc2 = QuantumCircuit(1)
    state2 = qc2.get_state()
    exp_val2 = expectation_value(state2, z_observable)
    logger.info(f"⟨0|Z|0⟩ = {exp_val2:.6f} (expected: 1.0)")
    logger.info("")


def main():
    """Run all quantum circuit examples."""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("QUANTUM CIRCUIT SIMULATOR - EXAMPLE DEMONSTRATIONS")
        logger.info("=" * 60 + "\n")
        
        # Run all examples
        example_bell_state()
        example_superposition()
        example_grover_search()
        example_phase_gates()
        example_multi_gate_circuit()
        example_expectation_value()
        
        logger.info("=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
