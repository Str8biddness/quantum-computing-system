"""Quantum Utilities Module

This module provides utility functions and classes for quantum computing:
- Visualization tools (Bloch sphere, state plotting, measurement histograms)
- Optimization techniques (circuit optimization, parameter optimization, gradient-based methods)
- Error mitigation and correction (ZNE, measurement error mitigation, noise models)

All utilities are designed to work seamlessly with the quantum simulator and algorithm modules.

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
from typing import List

# Import utility modules
try:
    from .visualization import QuantumVisualizer, visualize_state
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Visualization module not available: {e}")
    VISUALIZATION_AVAILABLE = False

try:
    from .optimization import (
        CircuitOptimizer, ParameterOptimizer, GradientOptimizer
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optimization module not available: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    from .error_mitigation import (
        MeasurementErrorMitigation, ZeroNoiseExtrapolation, NoiseModel, ProbabilisticErrorCancellation
    )
    ERROR_MITIGATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error mitigation module not available: {e}")
    ERROR_MITIGATION_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "Quantum Computing System"
__all__ = [
    # Visualization exports
    "QuantumVisualizer",
    "visualize_state",
    # Optimization exports
    "CircuitOptimizer",
    "ParameterOptimizer",
    "GradientOptimizer",
    # Error mitigation exports
    "MeasurementErrorMitigation",
    "ZeroNoiseExtrapolation",
    "NoiseModel",
    "ProbabilisticErrorCancellation",
    # Utility functions
    "get_available_utils",
]

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_utils() -> dict:
    """Get availability status of all utility modules.
    
    Returns:
        Dictionary mapping module names to availability status
        
    Example:
        >>> utils = get_available_utils()
        >>> if utils['visualization']:
        ...     print("Visualization available")
    """
    return {
        'visualization': VISUALIZATION_AVAILABLE,
        'optimization': OPTIMIZATION_AVAILABLE,
        'error_mitigation': ERROR_MITIGATION_AVAILABLE
    }


# Log module initialization
logger.info(f"Quantum Utilities module initialized (version {__version__})")
available = get_available_utils()
logger.info(f"Available utilities: {[k for k, v in available.items() if v]}")


if __name__ == "__main__":
    """Demonstration of utilities module."""
    print("=" * 60)
    print("QUANTUM UTILITIES MODULE")
    print("=" * 60)
    
    print("\nAvailable Utilities:")
    print("-" * 60)
    
    utils = get_available_utils()
    for util_name, available in utils.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{util_name.capitalize():20} {status}")
    
    print("\n" + "=" * 60)
    print(f"Total utilities available: {sum(utils.values())}/3")
    print("=" * 60)
