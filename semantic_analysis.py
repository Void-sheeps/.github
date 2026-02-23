import torch
import matplotlib.pyplot as plt
import numpy as np
from semantic_engine import SemanticEngine

def run_analysis():
    print("Running Analytic Semantic Engine Analysis...")
    model = SemanticEngine()

    # Cenário expandido para análise visual
    input_data = [
        ("MAE", 100.0),
        ("LIMITE", 50.0),
        ("CASA", 10.0),
        ("CASAS", 10.0),
        ("BARREIRA", 5.0),
        ("BARREIRAS", 5.0)
    ]

    words = [d[0] for d in input_data]
    freqs = [d[1] for d in input_data]

    output = model(words, freqs)

    centroids = np.array([item['centroid'] for item in output])
    intensities = [item['intensity'] for item in output]
    values = [item['value'] for item in output]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Geometria Analítica: Resultantes no Plano Cartesiano
    scatter = ax1.scatter(centroids[:, 1], centroids[:, 0], s=np.array(intensities)*100,
                         c=intensities, cmap='viridis', alpha=0.7)

    for i, txt in enumerate(words):
        ax1.annotate(txt, (centroids[i, 1], centroids[i, 0]), xytext=(8, 8),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    # Desenhar linhas de distância entre MAE e LIMITE
    p_mae = centroids[words.index("MAE")]
    p_limite = centroids[words.index("LIMITE")]
    ax1.plot([p_mae[1], p_limite[1]], [p_mae[0], p_limite[0]], 'r--', alpha=0.5, label='Distância Argumentativa')

    ax1.set_title("Analytic Semantic Geometry (Weighted Resultants)")
    ax1.set_xlabel("X (Cartesian)")
    ax1.set_ylabel("Y (Cartesian)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', lw=1)
    ax1.axvline(0, color='black', lw=1)
    fig.colorbar(scatter, ax=ax1, label='Logarithmic Intensity')
    ax1.legend()

    # 2. Distribuição de Magnitude Semântica
    bars = ax2.bar(words, values, color=plt.cm.magma(np.linspace(0.3, 0.8, len(words))))
    ax2.set_title("Semantic Magnitude (Lexical/Algebraic)")
    ax2.set_ylabel("Semantic Value")
    ax2.set_xlabel("Word Tokens")
    plt.xticks(rotation=45)

    # Adicionar valores no topo das barras
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("semantic_analysis.png")
    print("Analysis complete. Visualization saved to semantic_analysis.png")

if __name__ == "__main__":
    run_analysis()
