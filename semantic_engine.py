import torch
import torch.nn as nn
import argparse

class SemanticEngine(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. A GEOMETRIA (Camada de Embeddings / Matriz Latino-Românica) ---
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

        # Embedding (30 caracteres, dimensão 2)
        self.geometry = nn.Embedding(num_embeddings=30, embedding_dim=2)

        with torch.no_grad():
            self.geometry.weight.fill_(0.0) # Zera tudo
            # Coordenadas fixas baseadas no formulário
            self.set_coord('A', 0, 0)
            self.set_coord('B', 0, 1) # Barreira
            self.set_coord('C', 0, 2) # Casa/Caos
            self.set_coord('M', 0, 3) # Mãe
            self.set_coord('E', 4, 0) # E
            self.set_coord('L', 3, 3) # Limite (Hipotético)
            self.set_coord('I', 8, 0) # Limite (Hipotético)
            self.set_coord('S', 1, 8) # Sufixo plural

        # --- 2. A MEMÓRIA (Lexicon / Look-up Table) ---
        self.lexicon_values = {
            "BARREIRA": 2.0,
            "CASA": 5.0,
            "CAOS": 99.0,
            "CAIS": 8.0,
            "MAE": 100.0,
            "LIMITE": 50.0,
            "HELLO": 10.0,
            "LOREM": 15.0
        }

        # --- 3. A ÁLGEBRA (Operadores) ---
        self.algebraic_rules = {
            "S": lambda x: x * 2.0,
            "AO": lambda x: x * 10.0,
        }

    def set_coord(self, char, y, x):
        if char in self.char_to_idx:
            idx = self.char_to_idx[char]
            self.geometry.weight[idx] = torch.tensor([float(y), float(x)])

    def get_word_geometry(self, word):
        """Converte string -> Tensor de Coordenadas [N, 2]"""
        indices = [self.char_to_idx.get(c, 0) for c in word.upper() if c in self.char_to_idx]
        if not indices:
            return torch.zeros(1, 2)
        input_tensor = torch.tensor(indices, dtype=torch.long)
        return self.geometry(input_tensor)

    def log_intensity(self, frequency):
        """Aplica a Tabela de Logaritmos para medir densidade de informação."""
        return torch.log1p(torch.tensor(float(frequency), dtype=torch.float32))

    def forward(self, word_list, frequencies=None):
        """
        Processamento Híbrido com Intensidade Logarítmica e Resultante Cartesiana.
        """
        if frequencies is None:
            frequencies = [1.0] * len(word_list)

        results = []

        for word, freq in zip(word_list, frequencies):
            word = word.upper().replace('Ã', 'A').replace('Ê', 'E') # Normalização simples

            # Etapa A: Geometria (Projeção no Plano Cartesiano)
            coords = self.get_word_geometry(word)

            # Cálculo de Intensidade Logarítmica
            intensity = self.log_intensity(freq)

            # Resultante Central (Weighted Centroid)
            # Aqui aplicamos o peso da intensidade na resultante geométrica
            centroid = torch.mean(coords, dim=0) * intensity

            # Etapa B: Decisão Neuro-Simbólica
            final_value = 0.0
            method = "DESCONHECIDO"

            if word in self.lexicon_values:
                final_value = self.lexicon_values[word]
                method = "MEMÓRIA (Look-up)"
            else:
                for suffix, operation in self.algebraic_rules.items():
                    if word.endswith(suffix):
                        root = word[:-len(suffix)]
                        if root in self.lexicon_values:
                            base_val = self.lexicon_values[root]
                            t_base = torch.tensor(base_val, dtype=torch.float32)
                            t_final = operation(t_base)
                            final_value = t_final.item()
                            method = f"ÁLGEBRA (Raiz: {root})"
                            break

            results.append({
                "word": word,
                "centroid": centroid.tolist(),
                "intensity": intensity.item(),
                "value": final_value,
                "method": method
            })

        return results

def run_simulation():
    model = SemanticEngine()

    # Cenário: MÃE (Alta intensidade) vs LIMITE (Intensidade variável)
    input_data = [
        ("MAE", 100.0),
        ("LIMITE", 50.0),
        ("CASA", 10.0),
        ("CASAS", 10.0)
    ]

    words = [d[0] for d in input_data]
    freqs = [d[1] for d in input_data]

    output = model(words, freqs)

    print(f"{'PALAVRA':<10} | {'INTENS.':<8} | {'RESULTANTE (Y, X)':<20} | {'MÉTODO'}")
    print("-" * 70)

    for item in output:
        res_str = f"[{item['centroid'][0]:.2f}, {item['centroid'][1]:.2f}]"
        print(f"{item['word']:<10} | {item['intensity']:<8.4f} | {res_str:<20} | {item['method']}")

    # Cálculo de Distância (Qualidade Argumentativa)
    pos_mae = torch.tensor(next(i['centroid'] for i in output if i['word'] == 'MAE'))
    pos_limite = torch.tensor(next(i['centroid'] for i in output if i['word'] == 'LIMITE'))
    distancia = torch.norm(pos_mae - pos_limite)

    print(f"\nDistância Euclidiana (MAE vs LIMITE): {distancia.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytic Semantic Engine Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the engine simulation")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
