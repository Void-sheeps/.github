import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from epistemic_agent import State, EpistemicAgent, make_internalist_perturbations, make_externalist_perturbations, TerminalIterator

def run_analysis():
    print("Running Epistemic Agent Analysis...")

    corpus = [
        "python keywords include def class if else for while import return lambda",
        "list comprehension offers concise way to create lists in python",
        "decorators are functions that modify other functions or methods",
        "python uses duck typing and dynamic typing with strong typing",
        "context managers with statement handles resource acquisition and release",
        "generators use yield to create iterators efficiently",
        "f strings provide formatted string literals since python 3.6",
        "type hints introduced in pep 484 improve code readability",
        "asyncio enables concurrent code using async and await keywords",
        "walrus operator assigns values inside expressions using colon equals",
        "photosynthesis converts light energy into chemical energy stored in glucose",
        "the roman empire collapsed due to military overextension and economic strain",
        "sonata form consists of exposition development and recapitulation sections",
        "supply and demand curves intersect at the market equilibrium price",
        "plate tectonics describes the motion of lithospheric plates over the mantle",
        "the immune system distinguishes self from non-self through antigen recognition",
        "impressionist painters used loose brushwork to capture transient light effects",
        "neural networks learn by adjusting weights through gradient descent",
        "constitutional law governs the relationship between state power and individual rights",
        "thermodynamics describes energy transfer through heat work and entropy",
    ]

    python_seeds = [
        "def and class are fundamental keywords for defining functions and classes",
        "list comprehensions provide a pythonic way to transform iterables",
        "the with statement is used for context management and resource handling",
        "lambda creates anonymous functions for short single expression logic",
        "import and from are used to bring modules and names into namespace",
        "async def defines coroutine functions for asynchronous execution",
        "yield keyword turns a regular function into a generator iterator",
        "f strings are the modern preferred way to format strings in python",
        "type hints annotate variables and function signatures for static analysis",
        "walrus operator allows assignment within expressions using colon equals syntax",
    ]

    internalist = make_internalist_perturbations()
    externalist = make_externalist_perturbations(corpus)
    perturbations = internalist + externalist

    all_phi_history = []
    all_stab_history = []

    plt.figure(figsize=(14, 10))

    for i, seed in enumerate(python_seeds):
        agent = EpistemicAgent(perturbations)
        s0 = State(seed)
        it = TerminalIterator(agent, s0)

        phi_history = [s0.phi]
        stab_history = [s0.stability]

        for s in it:
            phi_history.append(s.phi)
            stab_history.append(s.stability)

        all_phi_history.append(phi_history)
        all_stab_history.append(stab_history)

        steps = range(len(phi_history))
        plt.subplot(2, 1, 1)
        plt.plot(steps, phi_history, alpha=0.6, label=f"Seed {i}" if i < 3 else "")

        plt.subplot(2, 1, 2)
        plt.plot(steps, stab_history, alpha=0.6, label=f"Seed {i}" if i < 3 else "")

    plt.subplot(2, 1, 1)
    plt.title("Epistemic Field Evolution: Phi (φ)")
    plt.ylabel("Phi")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Epistemic Field Evolution: Stability")
    plt.ylabel("Stability")
    plt.xlabel("Perturbation Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("epistemic_field_analysis.png")
    print("Analysis complete. Visualization saved to epistemic_field_analysis.png")

    # Final agent report for the last run
    print("\nFinal Agent Report (Last Seed):")
    rep = agent.report()
    for k, v in rep.items():
        print(f"  {k:>26}: {v}")

if __name__ == "__main__":
    run_analysis()
