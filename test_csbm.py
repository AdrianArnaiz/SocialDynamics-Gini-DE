"""
Test script to visualize various CSBM generation possibilities.
Demonstrates different parameter configurations with small networks (~50 nodes).
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from gdrq_1 import generate_csbm


def plot_csbm_graph(data, title="CSBM", color_by="labels", ax=None, pos=None):
    """
    Visualize a CSBM graph with networkx.
    
    Parameters
    ----------
    data : CSBMData
        The generated CSBM data
    title : str
        Plot title
    color_by : str
        'labels' or 'features' - what to use for node colors
    ax : matplotlib axis
        Axis to plot on (if None, creates new figure)
    pos : dict or None
        Pre-computed node positions (if None, compute new layout)
    
    Returns
    -------
    ax : matplotlib axis
        The axis with the plot
    pos : dict
        Node positions for reuse
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create networkx graph from adjacency
    G = nx.from_numpy_array(data.W)
    
    # Choose colors based on labels or features
    if color_by == "labels":
        node_colors = data.labels
        cmap = cm.RdBu_r
        vmin, vmax = -1, 1
        color_label = "Labels"
    else:  # features
        node_colors = data.features
        cmap = cm.RdYlGn
        vmin = data.features.min()
        vmax = data.features.max()
        color_label = "Features"
    
    # Use spring layout for better visualization (or reuse provided layout)
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(len(G.nodes())))
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=300,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax, label=color_label)
    
    # Add title with network stats
    n_edges = G.number_of_edges()
    density = nx.density(G)
    ax.set_title(f"{title}\n{len(G.nodes())} nodes, {n_edges} edges, density={density:.3f}")
    ax.axis('off')
    
    return ax, pos


def test_1_assortative_structure():
    """Test 1: Assortative Community Structure"""
    print("\n" + "="*60)
    print("Test 1: Assortative Community Structure")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Strong within-block, weak between-block connections
    data = generate_csbm(n=50, p_in=0.15, p_out=0.03, mu=1.0, sigma=1.0, seed=0)
    
    print(f"Blocks: {np.bincount(data.blocks)}")
    print(f"Labels: {np.unique(data.labels, return_counts=True)}")
    print(f"Features: mean={data.features.mean():.3f}, std={data.features.std():.3f}")
    
    # Use same layout for both plots
    _, pos = plot_csbm_graph(data, "Colored by Labels", color_by="labels", ax=axes[0])
    plot_csbm_graph(data, "Colored by Features", color_by="features", ax=axes[1], pos=pos)
    
    plt.tight_layout()
    plt.savefig("test_csbm_1_assortative.png", dpi=150, bbox_inches='tight')
    print("Saved: test_csbm_1_assortative.png")
    plt.close()


def test_2_signal_strength_variation():
    """Test 2: Signal Strength Variation"""
    print("\n" + "="*60)
    print("Test 2: Signal Strength Variation (mu)")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    mu_values = [0.2, 0.8, 2.0, 5.0]
    
    for i, mu in enumerate(mu_values):
        data = generate_csbm(n=50, p_in=0.15, p_out=0.03, mu=mu, sigma=1.0, seed=i)
        
        # Calculate correlation between features and labels
        corr = np.corrcoef(data.features, data.labels)[0, 1]
        
        print(f"\nmu={mu:.1f}:")
        print(f"  Feature-label correlation: {corr:.3f}")
        print(f"  Feature range: [{data.features.min():.2f}, {data.features.max():.2f}]")
        
        plot_csbm_graph(
            data, 
            f"mu={mu:.1f} (corr={corr:.3f})", 
            color_by="features",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.savefig("test_csbm_2_signal_strength.png", dpi=150, bbox_inches='tight')
    print("\nSaved: test_csbm_2_signal_strength.png")
    plt.close()


def test_3_noise_level_control():
    """Test 3: Noise Level Control"""
    print("\n" + "="*60)
    print("Test 3: Noise Level Control (sigma)")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    sigma_values = [0.3, 1.0, 2.0, 4.0]
    
    for i, sigma in enumerate(sigma_values):
        data = generate_csbm(n=50, p_in=0.15, p_out=0.03, mu=1.5, sigma=sigma, seed=i)
        
        # Calculate correlation between features and labels
        corr = np.corrcoef(data.features, data.labels)[0, 1]
        
        print(f"\nsigma={sigma:.1f}:")
        print(f"  Feature-label correlation: {corr:.3f}")
        print(f"  Feature std: {data.features.std():.2f}")
        
        plot_csbm_graph(
            data, 
            f"sigma={sigma:.1f} (corr={corr:.3f})", 
            color_by="features",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.savefig("test_csbm_3_noise_level.png", dpi=150, bbox_inches='tight')
    print("\nSaved: test_csbm_3_noise_level.png")
    plt.close()


def test_4_graph_separability():
    """Test 4: Graph Separability"""
    print("\n" + "="*60)
    print("Test 4: Graph Separability (p_in/p_out ratio)")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    configs = [
        {"p_in": 0.25, "p_out": 0.01, "label": "Easy (ratio=25)"},
        {"p_in": 0.20, "p_out": 0.05, "label": "Medium (ratio=4)"},
        {"p_in": 0.15, "p_out": 0.10, "label": "Hard (ratio=1.5)"},
        {"p_in": 0.12, "p_out": 0.12, "label": "No structure (ratio=1)"},
    ]
    
    for i, config in enumerate(configs):
        data = generate_csbm(
            n=50, 
            p_in=config["p_in"], 
            p_out=config["p_out"], 
            mu=1.0, 
            sigma=1.0, 
            seed=i
        )
        
        # Calculate modularity-like measure
        W = data.W
        m = W.sum() / 2  # total edges
        block_0 = (data.blocks == 0)
        block_1 = (data.blocks == 1)
        
        e_in = W[np.ix_(block_0, block_0)].sum() + W[np.ix_(block_1, block_1)].sum()
        e_out = W[np.ix_(block_0, block_1)].sum()
        
        print(f"\n{config['label']}:")
        print(f"  p_in={config['p_in']:.2f}, p_out={config['p_out']:.2f}")
        print(f"  Within-block edges: {e_in/2:.0f}, Between-block edges: {e_out:.0f}")
        print(f"  Modularity measure: {(e_in - e_out) / (e_in + e_out):.3f}")
        
        plot_csbm_graph(
            data, 
            config['label'], 
            color_by="labels",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.savefig("test_csbm_4_separability.png", dpi=150, bbox_inches='tight')
    print("\nSaved: test_csbm_4_separability.png")
    plt.close()


def test_5_balanced_vs_imbalanced():
    """Test 5: Balanced vs. Imbalanced Communities"""
    print("\n" + "="*60)
    print("Test 5: Balanced vs. Imbalanced Communities")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Balanced
    data_balanced = generate_csbm(
        n=50, p_in=0.15, p_out=0.03, mu=1.0, sigma=1.0, 
        balanced=True, seed=0
    )
    
    print("\nBalanced:")
    print(f"  Block sizes: {np.bincount(data_balanced.blocks)}")
    
    plot_csbm_graph(data_balanced, "Balanced Communities", color_by="labels", ax=axes[0])
    
    # Imbalanced (multiple runs to show variation)
    data_imbalanced = generate_csbm(
        n=50, p_in=0.15, p_out=0.03, mu=1.0, sigma=1.0, 
        balanced=False, seed=42
    )
    
    print("\nImbalanced (random):")
    print(f"  Block sizes: {np.bincount(data_imbalanced.blocks)}")
    
    plot_csbm_graph(data_imbalanced, "Imbalanced Communities", color_by="labels", ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("test_csbm_5_balanced.png", dpi=150, bbox_inches='tight')
    print("\nSaved: test_csbm_5_balanced.png")
    plt.close()


def test_6_combined_scenarios():
    """Test 6: Combined Scenarios (Challenging Cases)"""
    print("\n" + "="*60)
    print("Test 6: Combined Scenarios")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    scenarios = [
        {
            "name": "Easy: Strong graph + Strong features",
            "params": {"p_in": 0.25, "p_out": 0.02, "mu": 2.0, "sigma": 0.5},
        },
        {
            "name": "Graph only: Strong graph + Weak features",
            "params": {"p_in": 0.25, "p_out": 0.02, "mu": 0.3, "sigma": 2.0},
        },
        {
            "name": "Features only: Weak graph + Strong features",
            "params": {"p_in": 0.12, "p_out": 0.11, "mu": 2.5, "sigma": 0.5},
        },
        {
            "name": "Hard: Weak graph + Weak features",
            "params": {"p_in": 0.12, "p_out": 0.10, "mu": 0.4, "sigma": 2.0},
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        data = generate_csbm(n=50, seed=i, **scenario["params"])
        
        corr = np.corrcoef(data.features, data.labels)[0, 1]
        
        print(f"\n{scenario['name']}:")
        print(f"  Params: {scenario['params']}")
        print(f"  Feature-label correlation: {corr:.3f}")
        
        plot_csbm_graph(
            data, 
            f"{scenario['name']}\n(feat-label corr={corr:.2f})", 
            color_by="features",
            ax=axes[i]
        )
    
    plt.tight_layout()
    plt.savefig("test_csbm_6_combined.png", dpi=150, bbox_inches='tight')
    print("\nSaved: test_csbm_6_combined.png")
    plt.close()


def main():
    """Run all CSBM tests"""
    print("="*60)
    print("CSBM Generation Tests")
    print("Testing all model generation possibilities")
    print("="*60)
    
    # Run all tests
    test_1_assortative_structure()
    test_2_signal_strength_variation()
    test_3_noise_level_control()
    test_4_graph_separability()
    test_5_balanced_vs_imbalanced()
    test_6_combined_scenarios()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("Generated 6 visualization files.")
    print("="*60)


if __name__ == "__main__":
    main()
