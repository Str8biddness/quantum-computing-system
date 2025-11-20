"""Quantum Algorithms Module

This module provides implementations of fundamental quantum algorithms including:
- Grover's Search Algorithm: Quadratic speedup for unstructured search problems
- Quantum Fourier Transform (QFT): Foundation for many quantum algorithms
- Shor's Algorithm: Efficient integer factorization for cryptography
- Variational Quantum Eigensolver (VQE): Hybrid quantum-classical optimization

Each algorithm is implemented with comprehensive error handling, logging, and
example functions demonstrating usage patterns.

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
from typing import List, Dict, Any, Optional

# Import algorithm implementations
try:
    from .grovers import GroverSearch
    GROVERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Grover's algorithm not available: {e}")
    GROVERS_AVAILABLE = False

try:
    from .qft import QFT
    QFT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"QFT not available: {e}")
    QFT_AVAILABLE = False

try:
    from .shors import ShorsAlgorithm
    SHORS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Shor's algorithm not available: {e}")
    SHORS_AVAILABLE = False

try:
    from .vqe import VQE
    VQE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VQE not available: {e}")
    VQE_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "Quantum Computing System"
__all__ = [
    "GroverSearch",
    "QFT",
    "ShorsAlgorithm",
    "VQE",
    "get_available_algorithms",
    "get_algorithm_info",
]

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_algorithms() -> Dict[str, bool]:
    """Get availability status of all quantum algorithms.
    
    Returns:
        Dict[str, bool]: Dictionary mapping algorithm names to availability status
        
    Example:
        >>> available = get_available_algorithms()
        >>> print(f"Grover's available: {available['grovers']}")
    """
    return {
        'grovers': GROVERS_AVAILABLE,
        'qft': QFT_AVAILABLE,
        'shors': SHORS_AVAILABLE,
        'vqe': VQE_AVAILABLE
    }


def get_algorithm_info(algorithm_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific algorithm.
    
    Args:
        algorithm_name: Name of the algorithm ('grovers', 'qft', 'shors', 'vqe')
        
    Returns:
        Optional[Dict[str, Any]]: Algorithm information dictionary or None if not found
        
    Example:
        >>> info = get_algorithm_info('grovers')
        >>> print(info['description'])
    """
    algorithm_info = {
        'grovers': {
            'name': "Grover's Search Algorithm",
            'description': 'Quantum search algorithm providing quadratic speedup',
            'complexity': 'O(√N) vs classical O(N)',
            'use_cases': ['Database search', 'Optimization problems', 'Cryptanalysis'],
            'available': GROVERS_AVAILABLE
        },
        'qft': {
            'name': 'Quantum Fourier Transform',
            'description': 'Quantum analog of discrete Fourier transform',
            'complexity': 'O((log N)²) vs classical O(N log N)',
            'use_cases': ['Phase estimation', 'Period finding', "Shor's algorithm"],
            'available': QFT_AVAILABLE
        },
        'shors': {
            'name': "Shor's Factoring Algorithm",
            'description': 'Efficient quantum algorithm for integer factorization',
            'complexity': 'O((log N)³) vs classical exponential',
            'use_cases': ['Integer factorization', 'RSA cryptography breaking', 'Number theory'],
            'available': SHORS_AVAILABLE
        },
        'vqe': {
            'name': 'Variational Quantum Eigensolver',
            'description': 'Hybrid quantum-classical algorithm for finding ground states',
            'complexity': 'Depends on ansatz and classical optimizer',
            'use_cases': ['Molecular simulation', 'Optimization', 'Machine learning'],
            'available': VQE_AVAILABLE
        }
    }
    
    return algorithm_info.get(algorithm_name.lower())


def list_algorithms() -> List[str]:
    """Get list of all available algorithm names.
    
    Returns:
        List[str]: List of algorithm names that are currently available
        
    Example:
        >>> algorithms = list_algorithms()
        >>> for algo in algorithms:
        ...     print(f"Available: {algo}")
    """
    available = get_available_algorithms()
    return [name for name, is_available in available.items() if is_available]


# Log module initialization
logger.info(f"Quantum Algorithms module initialized (version {__version__})")
logger.info(f"Available algorithms: {list_algorithms()}")


if __name__ == "__main__":
    """Demonstration of algorithms module functionality."""
    print("=" * 60)
    print("QUANTUM ALGORITHMS MODULE")
    print("=" * 60)
    
    # Display available algorithms
    print("\nAvailable Algorithms:")
    print("-" * 60)
    for algo_name in ['grovers', 'qft', 'shors', 'vqe']:
        info = get_algorithm_info(algo_name)
        if info and info['available']:
            print(f"\n✓ {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Complexity: {info['complexity']}")
            print(f"  Use Cases: {', '.join(info['use_cases'])}")
        else:
            print(f"\n✗ {algo_name.upper()} - Not Available")
    
    # Summary
    print("\n" + "=" * 60)
    available_count = len(list_algorithms())
    print(f"Total Available: {available_count}/4 algorithms")
    print("=" * 60)
