import torch
import matplotlib.pyplot as plt
import seaborn as sns
from symbolic_attention import SymbolicAttentionEngine

def run_analysis():
    print("Running Refined Symbolic Attention Analysis...")

    # Valores de N para comparar: Identidade (1.0), Proporção Áurea (1.618), Dualidade (2.0), Transcendental (3.141)
    n_values = [1.0, 1.618, 2.0, 3.141]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for i, n_val in enumerate(n_values):
        # Inicializa a engine com o valor de N específico
        engine = SymbolicAttentionEngine(n_value=n_val)
        sequence = engine.vocab

        # Processa a sequência completa do vocabulário (5 tokens)
        (output, weights), x = engine.process_sequence(sequence, return_weights=True)

        # Converter pesos para numpy para visualização
        attn_weights = weights.detach().numpy()

        # Plotar heatmap de pesos de atenção
        sns.heatmap(attn_weights, annot=True, fmt=".3f",
                    xticklabels=[f"T{j}" for j in range(len(sequence))],
                    yticklabels=[f"T{j}" for j in range(len(sequence))],
                    ax=axes[i], cmap="magma")

        axes[i].set_title(f"Attention Weights Heatmap (N={n_val})")
        axes[i].set_xlabel("Key Tokens (Context)")
        axes[i].set_ylabel("Query Tokens (Focus)")

        # Adicionar legenda sobre o que cada T representa
        if i == 0:
            legend_text = "\n".join([f"T{j}: {tok[:20]}..." for j, tok in enumerate(sequence)])
            axes[i].text(1.05, 0.5, legend_text, transform=axes[i].transAxes,
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.suptitle("Symbolic N Attention (v2): Refined Interference Patterns", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Salvar o gráfico
    output_path = "symbolic_attention_analysis.png"
    plt.savefig(output_path)
    print(f"Analysis complete. Visualization saved to {output_path}")

if __name__ == "__main__":
    run_analysis()
