"""Quantum State and Circuit Visualization Module

Provides visualization utilities for quantum states, circuits, and measurement results.
Supports multiple visualization types including:
- Bloch sphere representations
- State vector bar charts
- Probability distributions
- Circuit diagrams
- Measurement histograms

Dependencies: matplotlib, numpy

Author: Quantum Computing System
Version: 1.0.0
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumVisualizer:
    """Main visualization class for quantum computing data."""
    
    def __init__(self, style: str = 'default'):
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use ('default', 'seaborn', 'ggplot')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")
        
        self.style = style
        if style != 'default':
            try:
                plt.style.use(style)
            except:
                logger.warning(f"Style '{style}' not available, using default")
        
        logger.info("QuantumVisualizer initialized")
    
    def plot_state_vector(self, 
                          state_vector: np.ndarray, 
                          title: str = "Quantum State Vector",
                          show_phase: bool = True) -> None:
        """Plot quantum state vector amplitudes and phases.
        
        Args:
            state_vector: Complex amplitudes of quantum state
            title: Plot title
            show_phase: Whether to show phase information
            
        Example:
            >>> visualizer = QuantumVisualizer()
            >>> state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
            >>> visualizer.plot_state_vector(state)
        """
        n_states = len(state_vector)
        amplitudes = np.abs(state_vector)
        phases = np.angle(state_vector)
        
        fig, axes = plt.subplots(1, 2 if show_phase else 1, figsize=(12, 5))
        if not show_phase:
            axes = [axes]
        
        # Plot amplitudes
        x = range(n_states)
        axes[0].bar(x, amplitudes, color='blue', alpha=0.7)
        axes[0].set_xlabel('Basis State')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Amplitudes')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'|{i}⟩' for i in x])
        axes[0].grid(True, alpha=0.3)
        
        # Plot phases
        if show_phase:
            axes[1].bar(x, phases, color='red', alpha=0.7)
            axes[1].set_xlabel('Basis State')
            axes[1].set_ylabel('Phase (radians)')
            axes[1].set_title(f'{title} - Phases')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f'|{i}⟩' for i in x])
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        logger.info(f"Plotted state vector with {n_states} basis states")
    
    def plot_bloch_sphere(self, 
                          state_vector: np.ndarray,
                          title: str = "Bloch Sphere") -> None:
        """Plot single qubit state on Bloch sphere.
        
        Args:
            state_vector: 2-element complex state vector
            title: Plot title
            
        Raises:
            ValueError: If state_vector is not 2-dimensional
        """
        if len(state_vector) != 2:
            raise ValueError("Bloch sphere only supports single qubit (2D state)")
        
        # Calculate Bloch vector components
        alpha, beta = state_vector[0], state_vector[1]
        
        # Bloch sphere coordinates
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2)
        
        # Draw axes
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='g', arrow_length_ratio=0.1, label='Y')
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='b', arrow_length_ratio=0.1, label='Z')
        
        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='black', arrow_length_ratio=0.15, 
                 linewidth=2, label='State')
        
        # Labels
        ax.text(1.3, 0, 0, '|+⟩', fontsize=12)
        ax.text(0, 1.3, 0, '|+i⟩', fontsize=12)
        ax.text(0, 0, 1.3, '|0⟩', fontsize=12)
        ax.text(0, 0, -1.3, '|1⟩', fontsize=12)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        plt.show()
        logger.info("Plotted state on Bloch sphere")
    
    def plot_measurement_results(self, 
                                 measurements: Dict[str, int],
                                 title: str = "Measurement Results") -> None:
        """Plot measurement outcome histogram.
        
        Args:
            measurements: Dictionary mapping basis states to counts
            title: Plot title
            
        Example:
            >>> results = {'00': 480, '11': 520}
            >>> visualizer.plot_measurement_results(results)
        """
        states = list(measurements.keys())
        counts = list(measurements.values())
        total = sum(counts)
        probabilities = [c / total for c in counts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot counts
        ax1.bar(range(len(states)), counts, color='purple', alpha=0.7)
        ax1.set_xlabel('Measured State')
        ax1.set_ylabel('Counts')
        ax1.set_title(f'{title} - Counts (Total: {total})')
        ax1.set_xticks(range(len(states)))
        ax1.set_xticklabels(states, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot probabilities
        ax2.bar(range(len(states)), probabilities, color='green', alpha=0.7)
        ax2.set_xlabel('Measured State')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'{title} - Probabilities')
        ax2.set_xticks(range(len(states)))
        ax2.set_xticklabels(states, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
        logger.info(f"Plotted measurement results for {len(states)} states")
    
    def plot_probability_distribution(self, 
                                      state_vector: np.ndarray,
                                      title: str = "Probability Distribution") -> None:
        """Plot probability distribution from state vector.
        
        Args:
            state_vector: Quantum state vector
            title: Plot title
        """
        probabilities = np.abs(state_vector) ** 2
        n_states = len(probabilities)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_states), probabilities, color='orange', alpha=0.7)
        plt.xlabel('Basis State')
        plt.ylabel('Probability')
        plt.title(title)
        plt.xticks(range(n_states), [f'|{i}⟩' for i in range(n_states)])
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Add probability values on bars
        for i, p in enumerate(probabilities):
            if p > 0.01:
                plt.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        logger.info(f"Plotted probability distribution for {n_states} states")


def visualize_state(state_vector: np.ndarray, 
                   visualization_type: str = 'state_vector') -> None:
    """Convenience function for quick state visualization.
    
    Args:
        state_vector: Quantum state to visualize
        visualization_type: Type of visualization ('state_vector', 'bloch', 'probability')
    """
    visualizer = QuantumVisualizer()
    
    if visualization_type == 'state_vector':
        visualizer.plot_state_vector(state_vector)
    elif visualization_type == 'bloch':
        visualizer.plot_bloch_sphere(state_vector)
    elif visualization_type == 'probability':
        visualizer.plot_probability_distribution(state_vector)
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")


if __name__ == "__main__":
    """Demonstration of visualization capabilities."""
    print("=" * 60)
    print("QUANTUM VISUALIZATION MODULE DEMO")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
    else:
        visualizer = QuantumVisualizer()
        
        # Example 1: Bell state
        print("\n1. Visualizing Bell state |00⟩ + |11⟩")
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        visualizer.plot_state_vector(bell_state, title="Bell State")
        visualizer.plot_probability_distribution(bell_state, title="Bell State Probabilities")
        
        # Example 2: Single qubit on Bloch sphere
        print("\n2. Visualizing |+⟩ state on Bloch sphere")
        plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        visualizer.plot_bloch_sphere(plus_state, title="|+⟩ State")
        
        # Example 3: Measurement results
        print("\n3. Visualizing measurement results")
        measurements = {
            '00': 245,
            '01': 12,
            '10': 8,
            '11': 235
        }
        visualizer.plot_measurement_results(measurements, title="Bell State Measurements")
        
        print("\n" + "=" * 60)
        print("Visualization demo complete!")
        print("=" * 60)
