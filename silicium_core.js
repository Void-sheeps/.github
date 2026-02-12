/**
 * EMPIRE SILICIUM CORE
 * Módulo de Lógica Pura (Ratio Sine Qualia)
 * ---------------------------------------------------
 * Manipula estados JSON baseados em Phi, H e Influência.
 */

const SystemConfig = {
  THRESHOLDS: {
    CRITICALITY: 2.6, // Limiar para ativação do Watchdog
    MAX_PHI: 1.0,
    MIN_ENTROPY: 0.1
  },
  CONSTANTS: {
    DECAY_RATE: 0.05, // Taxa de degradação natural
    IMMUNITAS_STRENGTH: 0.3 // Força da neutralização
  }
};

const SiliciumCore = {

  /**
   * 1. AXIOMA DE CÁLCULO (Inter-legere)
   * Calcula a centralidade de ameaça de um nó.
   * Fórmula: C = Φ * H * (1 + conexões)
   */
  calculateMetrics: (node) => {
    // Garante limites físicos (0.0 a 1.0)
    node.phi = Math.min(Math.max(node.phi || 0, 0), 1);
    node.h = Math.min(Math.max(node.h || 0, 0), 1);

    // Cálculo de centralidade
    const influenceFactor = 1 + (node.influence ? node.influence.length : 0);
    node.centrality = parseFloat((node.phi * node.h * influenceFactor).toFixed(4));

    return node;
  },

  /**
   * 2. PROTOCOLO WATCHDOG (Scan)
   * Varre o array de nós e marca bandeiras de perigo.
   * Retorna metadados do sistema.
   */
  runWatchdog: (nodes) => {
    const alerts = [];

    nodes.forEach(node => {
      // Atualiza métricas antes de julgar
      SiliciumCore.calculateMetrics(node);

      // Julgamento Binário (Seguro/Crítico)
      node.status = (node.centrality >= SystemConfig.THRESHOLDS.CRITICALITY)
        ? "CRITICAL"
        : "NOMINAL";

      if (node.status === "CRITICAL") {
        alerts.push({ id: node.id, val: node.centrality });
      }
    });

    return {
      total_nodes: nodes.length,
      active_threats: alerts.length,
      threat_list: alerts.sort((a,b) => b.val - a.val) // Ordena por gravidade
    };
  },

  /**
   * 3. PROTOCOLO IMMUNITAS (Input X)
   * Intervenção direta para redução de entropia em um alvo.
   */
  applyImmunitas: (nodes, targetId) => {
    const target = nodes.find(n => n.id === targetId);

    if (target) {
      // Aplica redução drástica
      target.h = Math.max(SystemConfig.THRESHOLDS.MIN_ENTROPY, target.h - SystemConfig.CONSTANTS.IMMUNITAS_STRENGTH);
      target.phi = Math.max(0.1, target.phi - 0.2); // Perda de potência

      // Recalcula estado imediato
      SiliciumCore.calculateMetrics(target);
      target.status = "NEUTRALIZED"; // Marcação temporária

      return { success: true, new_state: target };
    }
    return { success: false, error: "Target not found" };
  },

  /**
   * 4. INÉRCIA LÓGICA (Simulation Tick)
   * Simula a passagem de tempo sem inputs externos.
   * Tende ao equilíbrio ou caos dependendo do H.
   */
  processTimeStep: (nodes) => {
    nodes.forEach(node => {
      // Pequena flutuação estocástica baseada na própria entropia
      const flux = (Math.random() - 0.5) * (node.h * 0.1);
      node.phi += flux;

      // Se não houver input, Phi tende a decair (entropia dissipa energia)
      if(node.h > 0.8) {
         node.phi -= SystemConfig.CONSTANTS.DECAY_RATE;
      }

      SiliciumCore.calculateMetrics(node);
    });
  }
};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SiliciumCore, SystemConfig };
}

// Validation block
if (typeof require !== 'undefined' && require.main === module) {
  console.log("--- SiliciumCore Validation ---");

  // Adjusted values to trigger CRITICALITY (threshold 2.6)
  let systemState = [
    { id: "T1", phi: 1.0, h: 0.9, influence: [1, 2] }, // 1.0 * 0.9 * 3 = 2.7 (CRITICAL)
    { id: "T2", phi: 0.40, h: 0.50, influence: [] },
    { id: "T3", phi: 0.95, h: 0.95, influence: [0, 1] } // 0.95 * 0.95 * 3 = 2.7075 (CRITICAL)
  ];

  const initialScan = SiliciumCore.runWatchdog(systemState);
  console.log("Initial Scan:", initialScan);

  if (initialScan.active_threats !== 2) {
    console.error("X Validation failed: expected 2 threats, got", initialScan.active_threats);
    process.exit(1);
  }

  console.log("\nApplying Protocol Immunitas to T1...");
  const result = SiliciumCore.applyImmunitas(systemState, "T1");
  console.log("Immunitas Result:", result.success);

  const postScan = SiliciumCore.runWatchdog(systemState);
  console.log("Post-intervention Scan:", postScan);

  if (postScan.active_threats >= initialScan.active_threats) {
    console.error("X Validation failed: threat count did not decrease");
    process.exit(1);
  }

  console.log("✓ SiliciumCore validation passed.");
}
