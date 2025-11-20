"""Quantum Circuit and Parameter Optimization Module

Provides optimization utilities for quantum circuits and variational algorithms.
Includes:
- Circuit optimization and gate reduction
- Parameter optimization for variational algorithms
- Gradient-based and gradient-free optimizers
- Circuit depth reduction strategies
- Gate cancellation and commutation

Dependencies: numpy, scipy

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import warnings

# Optional optimization dependencies
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced optimization features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitOptimizer:
    """Optimizer for quantum circuit depth and gate count reduction."""
    
    def __init__(self):
        """Initialize circuit optimizer."""
        self.optimization_passes = []
        logger.info("CircuitOptimizer initialized")
    
    def cancel_adjacent_gates(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cancel adjacent inverse gate pairs.
        
        Args:
            gates: List of gate dictionaries with 'name', 'qubits', 'params'
            
        Returns:
            Optimized gate list with cancellations removed
        """
        optimized = []
        i = 0
        
        while i < len(gates):
            if i < len(gates) - 1:
                current = gates[i]
                next_gate = gates[i + 1]
                
                # Check if gates are inverses on same qubits
                if (self._are_inverse_gates(current, next_gate) and
                    current['qubits'] == next_gate['qubits']):
                    # Skip both gates
                    i += 2
                    logger.debug(f"Cancelled {current['name']} pair")
                    continue
            
            optimized.append(gates[i])
            i += 1
        
        logger.info(f"Gate cancellation: {len(gates)} -> {len(optimized)} gates")
        return optimized
    
    def _are_inverse_gates(self, gate1: Dict[str, Any], gate2: Dict[str, Any]) -> bool:
        """Check if two gates are inverses of each other."""
        inverse_pairs = [
            ('H', 'H'),
            ('X', 'X'),
            ('Y', 'Y'),
            ('Z', 'Z'),
            ('CNOT', 'CNOT'),
            ('RX', 'RX'),  # If params are opposite
            ('RY', 'RY'),
            ('RZ', 'RZ'),
        ]
        
        name1, name2 = gate1['name'], gate2['name']
        
        # Check basic inverse pairs
        if (name1, name2) in inverse_pairs:
            # For rotation gates, check if angles are opposite
            if name1 in ['RX', 'RY', 'RZ']:
                param1 = gate1.get('params', [0])[0]
                param2 = gate2.get('params', [0])[0]
                return np.isclose(param1, -param2)
            return True
        
        return False
    
    def commute_gates(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder gates using commutation rules to enable more cancellations.
        
        Args:
            gates: List of gates
            
        Returns:
            Reordered gate list
        """
        # Simple implementation - just return original for now
        # Full implementation would analyze qubit dependencies
        logger.info("Gate commutation pass (placeholder)")
        return gates
    
    def optimize_circuit(self, gates: List[Dict[str, Any]], 
                        passes: int = 3) -> List[Dict[str, Any]]:
        """Run multiple optimization passes on circuit.
        
        Args:
            gates: Initial gate list
            passes: Number of optimization passes
            
        Returns:
            Optimized gate list
        """
        optimized = gates
        
        for pass_num in range(passes):
            initial_count = len(optimized)
            optimized = self.cancel_adjacent_gates(optimized)
            optimized = self.commute_gates(optimized)
            
            if len(optimized) == initial_count:
                logger.info(f"Optimization converged after {pass_num + 1} passes")
                break
        
        reduction = len(gates) - len(optimized)
        logger.info(f"Total reduction: {reduction} gates ({len(gates)} -> {len(optimized)})")
        return optimized


class ParameterOptimizer:
    """Optimizer for variational algorithm parameters."""
    
    def __init__(self, method: str = 'COBYLA'):
        """Initialize parameter optimizer.
        
        Args:
            method: Optimization method ('COBYLA', 'BFGS', 'Nelder-Mead', 'Powell')
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for parameter optimization")
        
        self.method = method
        self.history = []
        logger.info(f"ParameterOptimizer initialized with method: {method}")
    
    def optimize(self, 
                cost_function: Callable[[np.ndarray], float],
                initial_params: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None,
                maxiter: int = 100) -> Dict[str, Any]:
        """Optimize parameters to minimize cost function.
        
        Args:
            cost_function: Function to minimize f(params) -> cost
            initial_params: Starting parameter values
            bounds: Optional parameter bounds [(min, max), ...]
            maxiter: Maximum iterations
            
        Returns:
            Dictionary with 'optimal_params', 'optimal_cost', 'iterations', 'success'
            
        Example:
            >>> def cost(params):
            ...     return np.sum(params**2)
            >>> optimizer = ParameterOptimizer()
            >>> result = optimizer.optimize(cost, np.array([1.0, 2.0]))
        """
        self.history = []
        
        def wrapped_cost(params):
            cost = cost_function(params)
            self.history.append({'params': params.copy(), 'cost': cost})
            return cost
        
        logger.info(f"Starting optimization with {len(initial_params)} parameters")
        
        result = minimize(
            wrapped_cost,
            initial_params,
            method=self.method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        logger.info(f"Optimization complete: {result.nit} iterations, "
                   f"final cost: {result.fun:.6f}")
        
        return {
            'optimal_params': result.x,
            'optimal_cost': result.fun,
            'iterations': result.nit,
            'success': result.success,
            'message': result.message,
            'history': self.history
        }
    
    def optimize_global(self,
                       cost_function: Callable[[np.ndarray], float],
                       bounds: List[Tuple[float, float]],
                       maxiter: int = 100) -> Dict[str, Any]:
        """Global optimization using differential evolution.
        
        Args:
            cost_function: Function to minimize
            bounds: Parameter bounds [(min, max), ...]
            maxiter: Maximum iterations
            
        Returns:
            Optimization result dictionary
        """
        logger.info("Starting global optimization (differential evolution)")
        
        result = differential_evolution(
            cost_function,
            bounds,
            maxiter=maxiter,
            seed=42
        )
        
        logger.info(f"Global optimization complete: final cost: {result.fun:.6f}")
        
        return {
            'optimal_params': result.x,
            'optimal_cost': result.fun,
            'iterations': result.nit,
            'success': result.success
        }


class GradientOptimizer:
    """Gradient-based optimization for quantum circuits."""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize gradient optimizer.
        
        Args:
            learning_rate: Step size for gradient descent
        """
        self.learning_rate = learning_rate
        self.history = []
        logger.info(f"GradientOptimizer initialized with lr={learning_rate}")
    
    def adam_optimize(self,
                     cost_function: Callable[[np.ndarray], float],
                     gradient_function: Callable[[np.ndarray], np.ndarray],
                     initial_params: np.ndarray,
                     epochs: int = 100,
                     beta1: float = 0.9,
                     beta2: float = 0.999,
                     epsilon: float = 1e-8) -> Dict[str, Any]:
        """Optimize using Adam algorithm.
        
        Args:
            cost_function: Function to minimize
            gradient_function: Function computing gradient
            initial_params: Starting parameters
            epochs: Number of optimization steps
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            
        Returns:
            Optimization result with optimal parameters
        """
        params = initial_params.copy()
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        
        self.history = []
        
        for epoch in range(epochs):
            # Compute gradient
            grad = gradient_function(params)
            
            # Update biased first and second moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected moments
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))
            
            # Update parameters
            params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Record history
            cost = cost_function(params)
            self.history.append({'epoch': epoch, 'cost': cost, 'params': params.copy()})
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: cost = {cost:.6f}")
        
        final_cost = cost_function(params)
        logger.info(f"Adam optimization complete: {epochs} epochs, "
                   f"final cost: {final_cost:.6f}")
        
        return {
            'optimal_params': params,
            'optimal_cost': final_cost,
            'epochs': epochs,
            'history': self.history
        }


if __name__ == "__main__":
    """Demonstration of optimization capabilities."""
    print("=" * 60)
    print("QUANTUM OPTIMIZATION MODULE DEMO")
    print("=" * 60)
    
    # Example 1: Circuit optimization
    print("\n1. Circuit Gate Optimization")
    optimizer = CircuitOptimizer()
    
    sample_gates = [
        {'name': 'H', 'qubits': [0], 'params': []},
        {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
        {'name': 'X', 'qubits': [1], 'params': []},
        {'name': 'X', 'qubits': [1], 'params': []},  # Can be cancelled
        {'name': 'RZ', 'qubits': [0], 'params': [0.5]},
        {'name': 'H', 'qubits': [0], 'params': []},  # Can be cancelled with first H
    ]
    
    optimized = optimizer.optimize_circuit(sample_gates)
    print(f"Original gates: {len(sample_gates)}")
    print(f"Optimized gates: {len(optimized)}")
    
    # Example 2: Parameter optimization
    if SCIPY_AVAILABLE:
        print("\n2. Parameter Optimization (Quadratic Function)")
        
        def quadratic_cost(params):
            return np.sum((params - np.array([1.0, 2.0, 3.0])) ** 2)
        
        param_opt = ParameterOptimizer(method='COBYLA')
        result = param_opt.optimize(
            quadratic_cost,
            initial_params=np.zeros(3),
            maxiter=50
        )
        
        print(f"Optimal parameters: {result['optimal_params']}")
        print(f"Optimal cost: {result['optimal_cost']:.6f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Success: {result['success']}")
    
    print("\n" + "=" * 60)
    print("Optimization demo complete!")
    print("=" * 60)
