import torch
from token_geometry import TokenGeometry
from semantic_engine import SemanticEngine

def run_context_analysis():
    print("="*60)
    print("CONTEXT ANALYSIS: RAW vs PRESET STATES")
    print("="*60)

    # 1. Geometry Analysis
    vocab_size = 20
    embedding_dim = 8
    geom_model = TokenGeometry(vocab_size, embedding_dim)

    # Define Preset Contexts
    hello_world = torch.tensor([1, 2])
    ipsum_lorem = torch.tensor([3, 4])

    geom_model.register_preset("HELLO_WORLD", hello_world)
    geom_model.register_preset("IPSUM_LOREM", ipsum_lorem)

    # Define Raw State
    raw_state = torch.tensor([10, 11])

    diff_hello = geom_model.compare_context_to_preset(raw_state, "HELLO_WORLD")
    diff_lorem = geom_model.compare_context_to_preset(raw_state, "IPSUM_LOREM")

    print(f"\n[GEOMETRY] Divergence Raw vs 'HELLO_WORLD': {diff_hello.item():.4f}")
    print(f"[GEOMETRY] Divergence Raw vs 'IPSUM_LOREM': {diff_lorem.item():.4f}")

    # 2. Semantic Engine Analysis
    sem_model = SemanticEngine()

    input_data = [
        ("HELLO", 1.0),
        ("LOREM", 1.0),
        ("RANDOM", 1.0)
    ]

    words = [d[0] for d in input_data]
    freqs = [d[1] for d in input_data]

    results = sem_model(words, freqs)

    print("\n[SEMANTIC] Context Values:")
    print(f"{'WORD':<10} | {'VALUE':<10} | {'METHOD'}")
    print("-" * 40)
    for item in results:
        print(f"{item['word']:<10} | {item['value']:<10.2f} | {item['method']}")

if __name__ == "__main__":
    run_context_analysis()
