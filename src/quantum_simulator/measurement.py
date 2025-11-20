"""Quantum Measurement Operations Module

Provides measurement functionality for quantum states.
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def measure_qubit(quantum_state, qubit_index: int, shots: int = 1) -> Dict[str, int]:
    """Measure a single qubit.
    
    Args:
        quantum_state: QuantumState object
        qubit_index: Index of qubit to measure
        shots: Number of measurement shots
        
    Returns:
        Dictionary with measurement outcomes and counts
    """
    if qubit_index < 0 or qubit_index >= quantum_state.num_qubits:
        raise ValueError(f"Qubit index must be in range [0, {quantum_state.num_qubits-1}]")
    
    results = {'0': 0, '1': 0}
    
    # Get probabilities for |0> and |1> on this qubit
    prob_0 = 0.0
    prob_1 = 0.0
    
    for basis_state in range(quantum_state.dimension):
        # Check if this qubit is 0 or 1 in this basis state
        qubit_value = (basis_state >> qubit_index) & 1
        probability = quantum_state.get_probability(basis_state)
        
        if qubit_value == 0:
            prob_0 += probability
        else:
            prob_1 += probability
    
    # Perform measurements
    for _ in range(shots):
        rand = np.random.random()
        if rand < prob_0:
            results['0'] += 1
        else:
            results['1'] += 1
    
    logger.debug(f"Measured qubit {qubit_index}: {results}")
    return results


def measure_all_qubits(quantum_state, shots: int = 1024) -> Dict[str, int]:
    """Measure all qubits in the quantum state.
    
    Args:
        quantum_state: QuantumState object
        shots: Number of measurement shots
        
    Returns:
        Dictionary mapping bitstrings to measurement counts
    """
    results = {}
    probabilities = quantum_state.get_probabilities()
    
    # Perform measurements according to probability distribution
    for _ in range(shots):
        # Sample from probability distribution
        rand = np.random.random()
        cumulative = 0.0
        
        for basis_state in range(quantum_state.dimension):
            cumulative += probabilities[basis_state]
            if rand < cumulative:
                # Convert basis state to bitstring
                bitstring = format(basis_state, f'0{quantum_state.num_qubits}b')
                results[bitstring] = results.get(bitstring, 0) + 1
                break
    
    logger.info(f"Measured all {quantum_state.num_qubits} qubits with {shots} shots")
    return results


def measure_and_collapse(quantum_state, qubit_index: int) -> int:
    """Measure a qubit and collapse the state.
    
    Args:
        quantum_state: QuantumState object
        qubit_index: Index of qubit to measure
        
    Returns:
        Measurement outcome (0 or 1)
    """
    if qubit_index < 0 or qubit_index >= quantum_state.num_qubits:
        raise ValueError(f"Qubit index must be in range [0, {quantum_state.num_qubits-1}]")
    
    # Calculate probabilities
    prob_0 = 0.0
    indices_0 = []
    indices_1 = []
    
    for basis_state in range(quantum_state.dimension):
        qubit_value = (basis_state >> qubit_index) & 1
        probability = quantum_state.get_probability(basis_state)
        
        if qubit_value == 0:
            prob_0 += probability
            indices_0.append(basis_state)
        else:
            indices_1.append(basis_state)
    
    # Measure
    outcome = 0 if np.random.random() < prob_0 else 1
    
    # Collapse state
    new_vector = np.zeros(quantum_state.dimension, dtype=complex)
    current_vector = quantum_state.get_vector()
    
    if outcome == 0:
        for idx in indices_0:
            new_vector[idx] = current_vector[idx]
        new_vector /= np.sqrt(prob_0) if prob_0 > 0 else 1
    else:
        for idx in indices_1:
            new_vector[idx] = current_vector[idx]
        prob_1 = 1 - prob_0
        new_vector /= np.sqrt(prob_1) if prob_1 > 0 else 1
    
    quantum_state.set_vector(new_vector)
    logger.debug(f"Measured qubit {qubit_index}, outcome: {outcome}, state collapsed")
    
    return outcome


def expectation_value(quantum_state, observable: np.ndarray) -> float:
    """Calculate expectation value of an observable.
    
    Args:
        quantum_state: QuantumState object
        observable: Hermitian matrix representing observable
        
    Returns:
        Expectation value (real number)
    """
    state_vector = quantum_state.get_vector()
    
    # <ψ|O|ψ>
    result = np.vdot(state_vector, observable @ state_vector)
    
    # Should be real for Hermitian observables
    expectation = np.real(result)
    
    logger.debug(f"Calculated expectation value: {expectation}")
    return expectation
