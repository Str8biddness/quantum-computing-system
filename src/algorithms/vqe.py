"""Variational Quantum Eigensolver (VQE) Implementation.

VQE is a hybrid quantum-classical algorithm for finding the ground state energy
of molecular Hamiltonians, with applications in quantum chemistry and materials science.

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import Callable, List, Optional, Tuple, Dict
import sys
import os
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_simulator import QuantumCircuit

logger = logging.getLogger(__name__)


class VQE:
    """Variational Quantum Eigensolver implementation.
    
    VQE uses a parameterized quantum circuit (ansatz) and classical optimization
    to find the minimum eigenvalue of a Hamiltonian operator.
    """
    
    def __init__(self, num_qubits: int, hamiltonian: np.ndarray):
        """Initialize VQE.
        
        Args:
            num_qubits: Number of qubits
            hamiltonian: Hamiltonian matrix (2^n x 2^n)
        
        Raises:
            ValueError: If hamiltonian dimensions don't match num_qubits
        """
        if hamiltonian.shape != (2**num_qubits, 2**num_qubits):
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} doesn't match "
                f"{num_qubits} qubits (expected {(2**num_qubits, 2**num_qubits)})"
            )
        
        self.num_qubits = num_qubits
        self.hamiltonian = hamiltonian
        self.dimension = 2 ** num_qubits
        
        # Verify Hamiltonian is Hermitian
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            logger.warning("Hamiltonian is not Hermitian!")
        
        # Calculate exact ground state energy for comparison
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        self.exact_ground_energy = np.min(eigenvalues)
        
        logger.info(f"Initialized VQE for {num_qubits} qubits")
        logger.info(f"Hamiltonian dimension: {self.dimension}x{self.dimension}")
        logger.info(f"Exact ground state energy: {self.exact_ground_energy:.6f}")
        
        # Optimization history
        self.history = {'energies': [], 'parameters': []}
    
    def create_ansatz(self, circuit: QuantumCircuit, parameters: np.ndarray) -> None:
        """Create parameterized quantum circuit (ansatz).
        
        Uses a hardware-efficient ansatz with rotation and entangling gates.
        
        Args:
            circuit: Quantum circuit to apply ansatz to
            parameters: Array of rotation angles
        """
        param_idx = 0
        
        # Layer 1: Initial rotations
        for qubit in range(self.num_qubits):
            if param_idx < len(parameters):
                circuit.apply_gate('RY', qubit, parameters[param_idx])
                param_idx += 1
        
        # Layer 2: Entangling layer
        for qubit in range(self.num_qubits - 1):
            circuit.apply_gate('CNOT', qubit, qubit + 1)
        
        # Layer 3: Rotation layer
        for qubit in range(self.num_qubits):
            if param_idx < len(parameters):
                circuit.apply_gate('RY', qubit, parameters[param_idx])
                param_idx += 1
    
    def measure_expectation(self, state_vector: np.ndarray) -> float:
        """Measure expectation value ⟨ψ|H|ψ⟩.
        
        Args:
            state_vector: Quantum state vector
        
        Returns:
            Expectation value of Hamiltonian
        """
        # ⟨ψ|H|ψ⟩ = ψ† H ψ
        expectation = np.real(state_vector.conj() @ self.hamiltonian @ state_vector)
        return expectation
    
    def cost_function(self, parameters: np.ndarray) -> float:
        """Cost function to minimize (expectation value).
        
        Args:
            parameters: Circuit parameters
        
        Returns:
            Energy expectation value
        """
        # Create circuit with current parameters
        circuit = QuantumCircuit(self.num_qubits)
        self.create_ansatz(circuit, parameters)
        
        # Get state vector
        state_vector = circuit.get_state().get_vector()
        
        # Calculate expectation value
        energy = self.measure_expectation(state_vector)
        
        # Store history
        self.history['energies'].append(energy)
        self.history['parameters'].append(parameters.copy())
        
        return energy
    
    def optimize(self, 
                 initial_parameters: Optional[np.ndarray] = None,
                 method: str = 'COBYLA',
                 max_iterations: int = 100) -> Dict:
        """Run VQE optimization.
        
        Args:
            initial_parameters: Starting parameters (random if None)
            method: Optimization method (COBYLA, SLSQP, etc.)
            max_iterations: Maximum optimization iterations
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("\n" + "=" * 60)
        logger.info("VQE OPTIMIZATION")
        logger.info("=" * 60)
        
        # Determine number of parameters needed
        num_params = 2 * self.num_qubits  # 2 rotation layers
        
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = np.random.uniform(0, 2*np.pi, num_params)
            logger.info(f"Initialized {num_params} random parameters")
        else:
            logger.info(f"Using provided {len(initial_parameters)} parameters")
        
        logger.info(f"Optimization method: {method}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Target (exact) energy: {self.exact_ground_energy:.6f}")
        
        # Reset history
        self.history = {'energies': [], 'parameters': []}
        
        # Run optimization
        logger.info("\nStarting optimization...")
        result = minimize(
            self.cost_function,
            initial_parameters,
            method=method,
            options={'maxiter': max_iterations}
        )
        
        optimal_energy = result.fun
        optimal_params = result.x
        
        # Calculate error
        error = abs(optimal_energy - self.exact_ground_energy)
        relative_error = error / abs(self.exact_ground_energy) * 100
        
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*60)
        logger.info(f"Converged: {result.success}")
        logger.info(f"Iterations: {len(self.history['energies'])}")
        logger.info(f"\nOptimal energy: {optimal_energy:.6f}")
        logger.info(f"Exact energy: {self.exact_ground_energy:.6f}")
        logger.info(f"Absolute error: {error:.6f}")
        logger.info(f"Relative error: {relative_error:.4f}%")
        
        return {
            'optimal_energy': optimal_energy,
            'exact_energy': self.exact_ground_energy,
            'optimal_parameters': optimal_params,
            'error': error,
            'relative_error': relative_error,
            'converged': result.success,
            'iterations': len(self.history['energies']),
            'history': self.history
        }
    
    def plot_convergence(self) -> None:
        """Display convergence information."""
        if not self.history['energies']:
            logger.warning("No optimization history available")
            return
        
        logger.info("\n" + "="*60)
        logger.info("CONVERGENCE ANALYSIS")
        logger.info("="*60)
        
        energies = self.history['energies']
        logger.info(f"\nTotal iterations: {len(energies)}")
        logger.info(f"Initial energy: {energies[0]:.6f}")
        logger.info(f"Final energy: {energies[-1]:.6f}")
        logger.info(f"Energy improvement: {energies[0] - energies[-1]:.6f}")
        
        # Show progression
        logger.info("\nEnergy progression (every 10 iterations):")
        for i in range(0, len(energies), 10):
            error = abs(energies[i] - self.exact_ground_energy)
            logger.info(f"  Iter {i:3d}: E = {energies[i]:.6f}, Error = {error:.6f}")


def create_h2_hamiltonian() -> np.ndarray:
    """Create H2 molecule Hamiltonian (simplified).
    
    Returns:
        2x2 Hamiltonian for H2 molecule
    """
    # Simplified H2 Hamiltonian in minimal basis
    H = np.array([
        [-1.0523, 0.3979],
        [0.3979, -0.4759]
    ])
    return H


def create_pauli_z_hamiltonian(num_qubits: int) -> np.ndarray:
    """Create Hamiltonian with Pauli-Z terms.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        Hamiltonian matrix
    """
    dim = 2 ** num_qubits
    H = np.diag([(-1)**bin(i).count('1') for i in range(dim)])
    return H


def example_h2_molecule():
    """Example: Find ground state of H2 molecule."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: H2 Molecule Ground State")
    logger.info("=" * 60)
    
    # Create H2 Hamiltonian
    H = create_h2_hamiltonian()
    
    # Run VQE
    vqe = VQE(num_qubits=1, hamiltonian=H)
    result = vqe.optimize(method='COBYLA', max_iterations=100)
    
    # Display convergence
    vqe.plot_convergence()


def example_multi_qubit():
    """Example: Multi-qubit VQE."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Multi-Qubit VQE")
    logger.info("=" * 60)
    
    # Create Pauli-Z Hamiltonian
    H = create_pauli_z_hamiltonian(2)
    
    logger.info("\nHamiltonian:")
    logger.info(H)
    
    # Run VQE
    vqe = VQE(num_qubits=2, hamiltonian=H)
    result = vqe.optimize(method='COBYLA', max_iterations=150)
    
    # Display convergence
    vqe.plot_convergence()


def example_parameter_study():
    """Example: Study effect of initial parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Initial Parameter Study")
    logger.info("=" * 60)
    
    H = create_h2_hamiltonian()
    
    # Try different initial parameters
    initial_params_list = [
        np.array([0.0, 0.0]),
        np.array([np.pi/4, np.pi/4]),
        np.array([np.pi/2, np.pi/2]),
        np.random.uniform(0, 2*np.pi, 2)
    ]
    
    names = ["zeros", "π/4", "π/2", "random"]
    
    for name, init_params in zip(names, initial_params_list):
        logger.info(f"\n--- Initial parameters: {name} ---")
        logger.info(f"Values: {init_params}")
        
        vqe = VQE(num_qubits=1, hamiltonian=H)
        result = vqe.optimize(
            initial_parameters=init_params,
            method='COBYLA',
            max_iterations=50
        )
        
        logger.info(f"Final energy: {result['optimal_energy']:.6f}")
        logger.info(f"Error: {result['error']:.6f}")
        logger.info(f"Iterations: {result['iterations']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    example_h2_molecule()
    example_multi_qubit()
    example_parameter_study()
