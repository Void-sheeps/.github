import torch
import torch.nn as nn
import torch.autograd as autograd
from torchdiffeq import odeint

class PhiLyapunovSentinel(nn.Module):
    def __init__(self, input_dim=4, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Constante crítica: 1/Phi (approx 0.618)
        # Em sistemas dinâmicos, muitas vezes representa o limiar de estabilidade
        self.phi_critical = 0.61803398

        # 1. Compressão do Estado Real
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Mish(), # Mish é suave e não-monotônica (melhor que ReLU para ODEs)
            nn.Linear(64, latent_dim)
        )

        # 2. Dinâmica do Campo (O "Clima" do Risco)
        self.field = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(), # Tanh é crucial para manter o campo limitado
            nn.Linear(64, latent_dim)
        )

        # 3. Projetor de Probabilidade
        self.risk_head = nn.Linear(latent_dim, 1)

    def ode_func(self, t, z):
        """
        Função envelopada para o solver.
        """
        return self.field(z)

    def compute_divergence(self, z, t):
        """
        Calcula a divergência do campo vetorial (nabla . f).
        Se > 0: O volume de fase expande (Caos/Explosão de Risco).
        Se < 0: O volume contrai (Estabilidade/Atrator Seguro).
        """
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.field(z)

            # Truque de Hutchinson para estimar o traço do Jacobiano (O(N))
            # Em vez de calcular a matriz Jacobiana inteira (O(N^2))
            e = torch.randn_like(z)
            e_dzdx = autograd.grad(dz_dt, z, e, create_graph=True)[0]
            divergence = (e_dzdx * e).sum(dim=1, keepdim=True)

        return divergence, dz_dt

    def forward(self, x, prediction_horizon=1.0):
        """
        x: Estado atual dos sensores
        prediction_horizon: Quão longe no futuro olhar
        """
        # Estado inicial latente
        z0 = self.encoder(x)

        # Tempo de integração
        t_span = torch.tensor([0, prediction_horizon]).float().to(x.device)

        # Integração da Trajetória
        # Usamos 'rk4' (Runge-Kutta 4) para precisão
        z_traj = odeint(self.ode_func, z0, t_span, method='rk4')
        z_final = z_traj[-1]

        # --- Análise de Bifurcação ---
        # Calculamos a estabilidade no ponto final previsto
        div, _ = self.compute_divergence(z_final, None)

        # Probabilidade base de incidente
        risk_logits = self.risk_head(z_final)
        base_prob = torch.sigmoid(risk_logits)

        # --- Modulação Phi ---
        # Se a divergência for alta (sistema instável), aumentamos o risco percebido
        # Se div > 0, stability_factor > 1 (Amplifica risco)
        stability_factor = torch.exp(div * self.phi_critical)

        adjusted_risk = torch.clamp(base_prob * stability_factor, 0, 1)

        return {
            "z_initial": z0,
            "z_final": z_final,
            "base_probability": base_prob,
            "divergence": div, # Métrica física de caos
            "adjusted_risk": adjusted_risk # Risco corrigido pela estabilidade
        }

if __name__ == "__main__":
    model = PhiLyapunovSentinel(input_dim=4)

    # Simulação: Sensores indicando vibração e temperatura subindo
    # Input: [Temp, Vibração, Pressão, Ruído]
    sensor_data = torch.tensor([[0.8, 0.9, 0.5, 0.2]])

    output = model(sensor_data, prediction_horizon=2.0)

    print("--- Diagnóstico Preditivo ---")
    print(f"Probabilidade Base (Neural): {output['base_probability'].item():.4f}")
    print(f"Divergência do Sistema (Lyapunov): {output['divergence'].item():.4f}")
    print(f"Risco Ajustado (Phi-Sentinela): {output['adjusted_risk'].item():.4f}")

    if output['adjusted_risk'] > 0.618: # Limiar Phi
        print("\nALERTA CRÍTICO: Colapso de Estabilidade Iminente.")
    else:
        print("\nStatus: Monitoramento Operacional.")
