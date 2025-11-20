"""Quantum Circuit Simulator Package.

This package provides a comprehensive quantum circuit simulator supporting
up to 32 qubits with state vector representation, standard quantum gates,
and measurement operations.

Modules:
    circuit: Quantum circuit construction and execution
    gates: Standard quantum gate implementations
    state: Quantum state management and operations
    measurement: Measurement and observation operations

Example:
    >>> from quantum_simulator import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.apply_gate('H', 0)
    >>> qc.apply_gate('CNOT', 0, 1)
    >>> results = qc.measure_all(shots=1000)
    >>> print(results)
"""

import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import main classes
try:
    from .circuit import QuantumCircuit
    from .gates import (
        QuantumGate,
        HadamardGate,
        PauliXGate,
        PauliYGate,
        PauliZGate,
        SGate,
        TGate,
        CNOTGate,
        ToffoliGate,
        RXGate,
        RYGate,
        RZGate,
        PhaseGate
    )
    from .state import QuantumState
    from .measurement import (
        measure_qubit,
        measure_all_qubits,
        measure_and_collapse,
        expectation_value
    )
    
    logger.info("Quantum simulator modules loaded successfully")
    
except ImportError as e:
    logger.error(f"Error importing quantum simulator modules: {e}")
    raise

# Package metadata
__version__ = "1.0.0"
__author__ = "Quantum Computing System"
__description__ = "A comprehensive quantum circuit simulator"

# Public API
__all__ = [
    # Main circuit class
    'QuantumCircuit',
    
    # State management
    'QuantumState',
    
    # Gate classes
    'QuantumGate',
    'HadamardGate',
    'PauliXGate',
    'PauliYGate',
    'PauliZGate',
    'SGate',
    'TGate',
    'CNOTGate',
    'ToffoliGate',
    'RXGate',
    'RYGate',
    'RZGate',
    'PhaseGate',
    
    # Measurement functions
    'measure_qubit',
    'measure_all_qubits',
    'measure_and_collapse',
    'expectation_value',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]

# Module initialization
logger.info(f"Quantum simulator package v{__version__} initialized")
logger.info(f"Supported operations: {len(__all__)} public functions/classes")

# Configuration constants
MAX_QUBITS = 32
DEFAULT_SHOTS = 1024
DEFAULT_TOLERANCE = 1e-10

logger.debug(f"Configuration: MAX_QUBITS={MAX_QUBITS}, DEFAULT_SHOTS={DEFAULT_SHOTS}")
