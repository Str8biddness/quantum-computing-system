#!/usr/bin/env python3
"""
GROVER'S QUANTUM SEARCH ALGORITHM
Implementation with optimal amplification and visualization
Author: Dakin Ellegood / Str8biddness
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

class GroverAlgorithm:
    def __init__(self, num_qubits=3):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.solution_state = None
        
    def create_oracle(self, target_state):
        """Create quantum oracle for target state"""
        qc = QuantumCircuit(self.num_qubits)
        
        # Mark the target state with phase inversion
        if target_state == '101':  # Example target
            qc.x(0)
            qc.x(2)
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
            qc.x(0)
            qc.x(2)
        elif target_state == '110':  # Another example
            qc.x(0)
            qc.x(1)
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
            qc.x(0)
            qc.x(1)
        else:
            # Generic oracle construction
            for i, bit in enumerate(reversed(target_state)):
                if bit == '0':
                    qc.x(i)
            
            qc.h(self.num_qubits - 1)
            qc.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)
            qc.h(self.num_qubits - 1)
            
            for i, bit in enumerate(reversed(target_state)):
                if bit == '0':
                    qc.x(i)
                    
        return qc
    
    def create_diffuser(self):
        """Create Grover diffuser for amplitude amplification"""
        qc = QuantumCircuit(self.num_qubits)
        
        # Apply H gates to all qubits
        qc.h(range(self.num_qubits))
        
        # Apply X gates to all qubits
        qc.x(range(self.num_qubits))
        
        # Apply multi-controlled Z gate
        qc.h(self.num_qubits - 1)
        qc.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        qc.h(self.num_qubits - 1)
        
        # Apply X gates to all qubits
        qc.x(range(self.num_qubits))
        
        # Apply H gates to all qubits
        qc.h(range(self.num_qubits))
        
        return qc
    
    def run_grover_search(self, target_state, iterations=None):
        """Execute complete Grover search algorithm"""
        if iterations is None:
            # Calculate optimal number of iterations
            iterations = int(np.pi/4 * np.sqrt(2**self.num_qubits))
            
        print(f"🔍 Grover Search: {self.num_qubits} qubits, target: {target_state}")
        print(f"📊 Optimal iterations: {iterations}")
        
        # Initialize quantum circuit
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Step 1: Create superposition
        qc.h(range(self.num_qubits))
        qc.barrier()
        
        # Step 2: Apply Grover iterations
        for _ in range(iterations):
            # Apply oracle
            oracle = self.create_oracle(target_state)
            qc.compose(oracle, inplace=True)
            qc.barrier()
            
            # Apply diffuser
            diffuser = self.create_diffuser()
            qc.compose(diffuser, inplace=True)
            qc.barrier()
        
        # Step 3: Measure all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        
        # Execute the circuit
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        
        return counts, qc
    
    def visualize_results(self, counts, title="Grover Search Results"):
        """Visualize measurement results"""
        # Sort results by probability
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(12, 6))
        plot_histogram(sorted_counts, title=title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/quantum/algorithms/grover_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print analysis
        print("\n📈 RESULT ANALYSIS:")
        for state, count in list(sorted_counts.items())[:5]:
            probability = count /
