import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from relational_field import (
    ConstraintResolver,
    Regime,
    setup_default_field,
)


def run_comparative_analysis():
    print("Running Comparative Relational Field Analysis...")

    field = setup_default_field()
    resolver = ConstraintResolver(field)

    results = []
    depths = range(1, 6)
    regimes = [Regime.STRICT, Regime.EXPANSIVE]

    for regime in regimes:
        for depth in depths:
            state = resolver.propagate(
                anchors={"absorption_identity": 1.0},
                target_node="justificative_displacement",
                depth=depth,
                contextual_gain=0.4,
                local_decay=1.0,
                regime=regime,
            )
            activation = state.active_nodes.get("justificative_displacement", 0.0)
            results.append(
                {
                    "Regime": regime.name,
                    "Depth": depth,
                    "Activation": activation,
                }
            )

    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="Regime", columns="Depth", values="Activation")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Activation of 'justificative_displacement' by Regime and Depth")
    plt.savefig("relational_analysis.png")
    print("Comparative analysis complete. Visualization saved to relational_analysis.png")


if __name__ == "__main__":
    run_comparative_analysis()
