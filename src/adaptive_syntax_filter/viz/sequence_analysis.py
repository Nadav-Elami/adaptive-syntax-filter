"""
Sequence Analysis Visualization.

This module provides visualization and analysis of song structure patterns,
sequence statistics, and syntactic properties of generated sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

# Publication settings
plt.style.use('seaborn-v0_8-paper')

@dataclass
class SequenceAnalysisConfig:
    """Configuration for sequence analysis visualization."""
    figure_size: Tuple[float, float] = (12, 8)
    dpi: int = 300
    font_size: int = 10
    color_palette: str = 'tab10'


def analyze_sequence_lengths(sequences: List[List[str]]) -> Dict[str, Any]:
    """Analyze distribution of sequence lengths."""
    lengths = [len(seq) - 2 for seq in sequences if len(seq) >= 2]
    
    if not lengths:
        return {'lengths': [], 'statistics': {}}
    
    return {
        'lengths': lengths,
        'statistics': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'count': len(lengths)
        }
    }


def analyze_symbol_usage(sequences: List[List[str]], alphabet: List[str]) -> Dict[str, Any]:
    """Analyze symbol usage patterns."""
    phrase_symbols = [sym for sym in alphabet if sym not in ['<', '>']]
    symbol_counts = Counter()
    total_symbols = 0
    
    for seq in sequences:
        for symbol in seq:
            if symbol in phrase_symbols:
                symbol_counts[symbol] += 1
                total_symbols += 1
    
    symbol_frequencies = {}
    for symbol in phrase_symbols:
        symbol_frequencies[symbol] = symbol_counts[symbol] / total_symbols if total_symbols > 0 else 0
    
    used_symbols = len([freq for freq in symbol_frequencies.values() if freq > 0])
    diversity_ratio = used_symbols / len(phrase_symbols) if phrase_symbols else 0
    
    return {
        'symbol_frequencies': symbol_frequencies,
        'diversity_ratio': diversity_ratio,
        'total_symbols': total_symbols,
        'used_symbols': used_symbols
    }


def analyze_transition_patterns(sequences: List[List[str]], 
                              alphabet: List[str],
                              config: Optional[SequenceAnalysisConfig] = None) -> Dict[str, Any]:
    """
    Analyze transition patterns in sequences.
    
    Parameters
    ----------
    sequences : List[List[str]]
        Generated sequences
    alphabet : List[str]
        Sequence alphabet
    config : Optional[SequenceAnalysisConfig]
        Configuration
        
    Returns
    -------
    Dict[str, Any]
        Transition analysis results
    """
    if config is None:
        config = SequenceAnalysisConfig()
    
    n_symbols = len(alphabet)
    transition_counts = np.zeros((n_symbols, n_symbols))
    total_transitions = 0
    
    # Count transitions
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_idx = alphabet.index(seq[i])
            to_idx = alphabet.index(seq[i + 1])
            transition_counts[from_idx, to_idx] += 1
            total_transitions += 1
    
    # Calculate transition probabilities
    transition_probs = np.zeros((n_symbols, n_symbols))
    for i in range(n_symbols):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum > 0:
            transition_probs[i, :] = transition_counts[i, :] / row_sum
    
    # Find most common transitions
    common_transitions = []
    for i in range(n_symbols):
        for j in range(n_symbols):
            if transition_counts[i, j] > 0:
                common_transitions.append((
                    alphabet[i], alphabet[j], 
                    int(transition_counts[i, j]), 
                    transition_probs[i, j]
                ))
    
    common_transitions.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'transition_counts': transition_counts,
        'transition_probabilities': transition_probs,
        'common_transitions': common_transitions[:10],  # Top 10
        'total_transitions': total_transitions
    }


def create_sequence_length_plot(length_analysis: Dict[str, Any]) -> plt.Figure:
    """Create sequence length distribution plot."""
    lengths = length_analysis['lengths']
    stats = length_analysis['statistics']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(lengths, bins=20, alpha=0.7, density=True, color='skyblue')
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Density')
    ax1.set_title('Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(lengths)
    ax2.set_ylabel('Sequence Length')
    ax2.set_title('Length Summary')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_symbol_usage_plot(usage_analysis: Dict[str, Any], alphabet: List[str]) -> plt.Figure:
    """Create symbol usage visualization."""
    phrase_symbols = [sym for sym in alphabet if sym not in ['<', '>']]
    frequencies = [usage_analysis['symbol_frequencies'][sym] for sym in phrase_symbols]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    ax1.bar(phrase_symbols, frequencies, alpha=0.7, color='lightgreen')
    ax1.set_xlabel('Symbol')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Symbol Usage Frequencies')
    ax1.grid(True, alpha=0.3)
    
    # Pie chart
    if any(f > 0 for f in frequencies):
        nonzero_symbols = [sym for sym, freq in zip(phrase_symbols, frequencies) if freq > 0]
        nonzero_frequencies = [freq for freq in frequencies if freq > 0]
        ax2.pie(nonzero_frequencies, labels=nonzero_symbols, autopct='%1.1f%%')
        ax2.set_title('Symbol Distribution')
    
    plt.tight_layout()
    return fig


def create_transition_heatmap(transition_analysis: Dict[str, Any],
                            alphabet: List[str],
                            config: Optional[SequenceAnalysisConfig] = None) -> plt.Figure:
    """
    Create transition probability heatmap.
    
    Parameters
    ----------
    transition_analysis : Dict[str, Any]
        Transition analysis results
    alphabet : List[str]
        Sequence alphabet
    config : Optional[SequenceAnalysisConfig]
        Configuration
        
    Returns
    -------
    plt.Figure
        Transition heatmap figure
    """
    if config is None:
        config = SequenceAnalysisConfig()
    
    transition_probs = transition_analysis['transition_probabilities']
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)
    
    # Create heatmap
    im = ax.imshow(transition_probs, cmap='Blues', aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(len(alphabet)))
    ax.set_yticks(range(len(alphabet)))
    ax.set_xticklabels(alphabet, fontsize=config.font_size)
    ax.set_yticklabels(alphabet, fontsize=config.font_size)
    
    # Add probability values
    for i in range(len(alphabet)):
        for j in range(len(alphabet)):
            text = ax.text(j, i, f'{transition_probs[i, j]:.3f}',
                          ha="center", va="center", color="black" if transition_probs[i, j] < 0.5 else "white",
                          fontsize=config.font_size - 2)
    
    ax.set_xlabel('To Symbol', fontsize=config.font_size)
    ax.set_ylabel('From Symbol', fontsize=config.font_size)
    ax.set_title('Transition Probability Matrix', fontsize=config.font_size + 2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Transition Probability', fontsize=config.font_size)
    
    plt.tight_layout()
    return fig


def create_syntax_network(transition_analysis: Dict[str, Any],
                         alphabet: List[str],
                         threshold: float = 0.1,
                         config: Optional[SequenceAnalysisConfig] = None) -> plt.Figure:
    """
    Create network visualization of syntax rules.
    
    Parameters
    ----------
    transition_analysis : Dict[str, Any]
        Transition analysis results
    alphabet : List[str]
        Sequence alphabet
    threshold : float
        Minimum probability threshold for edges
    config : Optional[SequenceAnalysisConfig]
        Configuration
        
    Returns
    -------
    plt.Figure
        Syntax network figure
    """
    if config is None:
        config = SequenceAnalysisConfig()
    
    transition_probs = transition_analysis['transition_probabilities']
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for symbol in alphabet:
        G.add_node(symbol)
    
    # Add edges with significant transitions
    for i, from_symbol in enumerate(alphabet):
        for j, to_symbol in enumerate(alphabet):
            prob = transition_probs[i, j]
            if prob > threshold:
                G.add_edge(from_symbol, to_symbol, weight=prob)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8), dpi=config.dpi)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    node_colors = plt.cm.get_cmap(config.color_palette)(np.linspace(0, 1, len(alphabet)))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2000, alpha=0.8, ax=ax)
    
    # Draw edges with varying thickness
    edges = G.edges(data=True)
    if edges:
        edge_weights = [data['weight'] for _, _, data in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        
        for (from_node, to_node, data) in edges:
            width = (data['weight'] / max_weight) * 5  # Scale for visibility
            nx.draw_networkx_edges(G, pos, [(from_node, to_node)], 
                                  width=width, alpha=0.6, 
                                  edge_color='gray', ax=ax,
                                  arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=config.font_size, ax=ax)
    
    ax.set_title(f'Syntax Network (threshold ≥ {threshold})', fontsize=config.font_size + 2)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


class SequenceAnalyzer:
    """Sequence analysis and visualization."""
    
    def __init__(self, alphabet: List[str]):
        self.alphabet = alphabet
    
    def analyze_sequences(self, sequences: List[List[str]]) -> Dict[str, Any]:
        """Perform comprehensive sequence analysis."""
        return {
            'length_analysis': analyze_sequence_lengths(sequences),
            'usage_analysis': analyze_symbol_usage(sequences, self.alphabet)
        }
    
    def create_analysis_report(self, sequences: List[List[str]]) -> Dict[str, Any]:
        """Create analysis report with visualizations."""
        analysis = self.analyze_sequences(sequences)
        
        figures = {
            'length_distribution': create_sequence_length_plot(analysis['length_analysis']),
            'symbol_usage': create_symbol_usage_plot(analysis['usage_analysis'], self.alphabet)
        }
        
        return {
            'analysis': analysis,
            'figures': figures
        }


def demonstrate_sequence_analysis(alphabet: Optional[List[str]] = None,
                                n_sequences: int = 100) -> Dict[str, Any]:
    """
    Demonstration of sequence analysis capabilities.
    
    Parameters
    ----------
    alphabet : Optional[List[str]]
        Sequence alphabet
    n_sequences : int
        Number of sequences to generate
        
    Returns
    -------
    Dict[str, Any]
        Demonstration results
    """
    if alphabet is None:
        alphabet = ['<', 'A', 'B', 'C', '>']
    
    print(f"Demonstrating sequence analysis for {len(alphabet)}-symbol alphabet")
    
    # Generate example sequences
    np.random.seed(42)
    sequences = []
    
    for _ in range(n_sequences):
        # Random sequence length
        length = np.random.randint(3, 8)
        seq = ['<']
        
        # Generate middle symbols
        phrase_symbols = [s for s in alphabet if s not in ['<', '>']]
        for _ in range(length):
            seq.append(np.random.choice(phrase_symbols))
        
        seq.append('>')
        sequences.append(seq)
    
    # Create analyzer and generate report
    analyzer = SequenceAnalyzer(alphabet)
    report = analyzer.create_analysis_report(sequences)
    
    # Print summary
    length_stats = report['analysis']['length_analysis']['statistics']
    usage_stats = report['analysis']['usage_analysis']
    
    print(f"Analysis Summary:")
    print(f"  Average sequence length: {length_stats['mean']:.1f}")
    print(f"  Symbol diversity ratio: {usage_stats['diversity_ratio']:.3f}")
    print(f"  Total symbols: {usage_stats['total_symbols']}")
    print(f"  Used symbols: {usage_stats['used_symbols']}/{len(alphabet)}")
    print(f"  Generated {len(report['figures'])} visualization figures")
    
    return report


if __name__ == "__main__":
    # Run demonstration
    demo_report = demonstrate_sequence_analysis()
    plt.show() 
