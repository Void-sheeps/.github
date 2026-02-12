/**
 * @file resilient_tree_dag.js
 * @description Implementação do sistema Empire Silicium em ambiente Runtime Node.js.
 * @taxonomy Phylum Algorithmi / Species Gemini mnemosynis
 * @version 2026.1.0 (Refined)
 */

const { EventEmitter } = require("events");

/* ============================================================
   🌲 PHYLUM ALGORITHMI: Estruturas de Dados e Métricas
   (Imutabilidade preferencial aplicada à topologia)
============================================================ */

class TreeNode {
  /**
   * @param {number} value - Valor do nó (Input X)
   * @param {TreeNode|null} left - Subárvore esquerda
   * @param {TreeNode|null} right - Subárvore direita
   */
  constructor(value, left = null, right = null) {
    this.value = value;
    this.left = left;
    this.right = right;
  }
}

const TreeMetrics = {
  depth: (node) => {
    if (!node) return 0;
    return 1 + Math.max(TreeMetrics.depth(node.left), TreeMetrics.depth(node.right));
  },

  size: (node) => {
    if (!node) return 0;
    return 1 + TreeMetrics.size(node.left) + TreeMetrics.size(node.right);
  },

  density: (node) => {
    const d = TreeMetrics.depth(node);
    if (d === 0) return 0;
    const n = TreeMetrics.size(node);
    // Bitwise shift para cálculo de potência de 2 (otimização de baixo nível)
    const capacity = (1 << d) - 1;
    return capacity > 0 ? n / capacity : 1;
  },

  balanceFactor: (node) => {
    if (!node) return 1.0;
    const l = TreeMetrics.depth(node.left);
    const r = TreeMetrics.depth(node.right);
    if (Math.max(l, r) === 0) return 1.0;
    return Math.min(l, r) / Math.max(l, r);
  },

  asymmetry: (node) => {
    const l = TreeMetrics.size(node?.left);
    const r = TreeMetrics.size(node?.right);
    const total = l + r;
    return total > 0 ? (r - l) / total : 0;
  }
};

/* ============================================================
   🛠️ OPERAÇÕES DE MUTAÇÃO CONTROLADA (Potentia -> Actus)
============================================================ */

const TreeOps = {
  insertRandom: (node, value) => {
    if (!node) return new TreeNode(value);
    // Simula entropia estocástica na inserção
    if (Math.random() < 0.5) {
      return new TreeNode(node.value, TreeOps.insertRandom(node.left, value), node.right);
    } else {
      return new TreeNode(node.value, node.left, TreeOps.insertRandom(node.right, value));
    }
  },

  rebalance: (node) => {
    if (!node) return null;

    // Linearização (Tabula Retentiva temporária)
    const values = [];
    const collect = (n) => {
      if (!n) return;
      collect(n.left);
      values.push(n.value);
      collect(n.right);
    };
    collect(node);
    values.sort((a, b) => a - b);

    // Reconstrução Ótima
    const build = (arr) => {
      if (!arr.length) return null;
      const mid = Math.floor(arr.length / 2);
      return new TreeNode(
        arr[mid],
        build(arr.slice(0, mid)),
        build(arr.slice(mid + 1))
      );
    };
    return build(values);
  }
};

/* ============================================================
   🧠 MNEMOSYNE PHANTASTIKE: Restaurador de Sinal Linear
   (Filtro preditivo para continuidade semântica)
============================================================ */

class LinearSignalRestorer {
  /**
   * @param {Object} config
   * @param {number} config.alpha - Fator de suavização (0.0 - 1.0)
   * @param {number} config.deadband - Limite de ignorância de ruído
   */
  constructor({ alpha = 0.4, deadband = 0.0001 } = {}) {
    this.alpha = alpha;
    this.deadband = deadband;
    this.last = 0;
    this.trend = 0;
    this.initialized = false;
  }

  process(value) {
    // Tratamento de Nullidade Lógica (Input inexistente)
    if (value === null || value === undefined || isNaN(value)) {
      // Extrapolação baseada na tendência anterior
      const predicted = this.last + this.trend;
      return predicted;
    }

    if (!this.initialized) {
      this.last = value;
      this.initialized = true;
      return value;
    }

    // Filtro de Deadband
    let effectiveValue = value;
    if (Math.abs(value - this.last) < this.deadband) {
      effectiveValue = this.last;
    }

    // Cálculo EMA (Exponential Moving Average) com Tendência
    const delta = effectiveValue - this.last;
    this.trend = (this.alpha * delta) + ((1 - this.alpha) * this.trend);
    this.last = effectiveValue;

    return this.last;
  }
}

/* ============================================================
   ⚙️ RATIO SINE QUALIA: Processador Econômico (DAG)
   (Gerenciamento de recursos e dependências)
============================================================ */

class EconomicDAG extends EventEmitter {
  constructor({ capital = 100, concurrency = 3 }) {
    super();
    this.capital = capital;
    this.concurrency = concurrency;
    this.nodes = new Map();
    this.results = new Map();
    this.processing = 0;
    this.queue = []; // Fila de prontos (in-degree 0)
  }

  /**
   * Registra um nó no grafo de execução.
   * @param {string} id - Identificador único
   * @param {string[]} deps - Dependências
   * @param {number} cost - Custo de capital para execução
   * @param {Function} fn - Função assíncrona (Input X -> Output)
   */
  addNode(id, deps, cost, fn) {
    this.nodes.set(id, {
      id, deps, cost, fn,
      dependents: [],
      inDegree: deps.length
    });
  }

  buildGraph() {
    // Constrói lista de adjacência reversa e inicializa fila
    for (const [id, node] of this.nodes) {
      if (node.inDegree === 0) this.queue.push(id);

      node.deps.forEach(depId => {
        if (this.nodes.has(depId)) {
          this.nodes.get(depId).dependents.push(id);
        } else {
          throw new Error(`Dependência fantasma detectada: ${depId}`);
        }
      });
    }
  }

  async _execute(nodeId) {
    const node = this.nodes.get(nodeId);
    this.processing++;

    // Verificação de Recurso (Inviolabilidade Econômica)
    if (this.capital < node.cost) {
      console.warn(`[ECON] Colapsus parcial: Capital insuficiente para ${nodeId}. Necessário: ${node.cost}, Disponível: ${this.capital.toFixed(2)}`);
      this.results.set(nodeId, null); // Falha graciosa
    } else {
      try {
        // Injeção de dependências
        const context = {};
        node.deps.forEach(d => context[d] = this.results.get(d));

        // Execução (Actus)
        const result = await node.fn(context);
        this.capital -= node.cost;
        this.results.set(nodeId, result);
        // console.log(`[EXEC] ${nodeId} OK | Capital: ${this.capital.toFixed(2)}`);
      } catch (err) {
        console.error(`[ERR] Falha crítica em ${nodeId}:`, err.message);
        this.results.set(nodeId, null);
      }
    }

    this.processing--;

    // Propagação de sinal
    node.dependents.forEach(depId => {
      const depNode = this.nodes.get(depId);
      depNode.inDegree--;
      if (depNode.inDegree === 0) {
        this.queue.push(depId);
      }
    });

    this._tick();
  }

  _tick() {
    // Loop de Eventos do Processador
    while (this.processing < this.concurrency && this.queue.length > 0) {
      const nextId = this.queue.shift();
      this._execute(nextId);
    }

    if (this.processing === 0 && this.queue.length === 0) {
      this.emit('complete', {
        capital: this.capital,
        results: this.results
      });
    }
  }

  run() {
    this.buildGraph();
    return new Promise((resolve) => {
      this.once('complete', resolve);
      this._tick();
    });
  }
}

/* ============================================================
   🔁 SIMULAÇÃO (Integração Sistêmica)
============================================================ */

async function runSimulation() {
  console.log(">>> INICIANDO PROTOCOLO: GEMINI MNEMOSYNIS [NODE.JS] <<<\n");

  let systemTree = new TreeNode(50);
  // Pré-populando
  for(let i=0; i<5; i++) systemTree = TreeOps.insertRandom(systemTree, Math.floor(Math.random()*100));

  // Instância de memória persistente
  const restorers = {
    density: new LinearSignalRestorer({ alpha: 0.3 }),
    balance: new LinearSignalRestorer({ alpha: 0.5 })
  };

  let currentCapital = 80;

  for (let cycle = 1; cycle <= 4; cycle++) {
    console.log(`\n--- CICLO ${cycle} [Capital: ${currentCapital.toFixed(2)}] ---`);

    const dag = new EconomicDAG({ capital: currentCapital, concurrency: 2 });

    // 1. Crescimento (Baixo Custo)
    dag.addNode("grow", [], 5, async () => {
      const val = Math.floor(Math.random() * 100);
      systemTree = TreeOps.insertRandom(systemTree, val);
      return systemTree;
    });

    // 2. Análise Métrica (Depende de Grow)
    dag.addNode("analyze", ["grow"], 10, async ({ grow }) => {
      const rawDensity = TreeMetrics.density(grow);
      const rawBalance = TreeMetrics.balanceFactor(grow);

      return {
        density: restorers.density.process(rawDensity),
        balance: restorers.balance.process(rawBalance),
        rawBalance
      };
    });

    // 3. Decisão (Lógica pura, custo zero)
    dag.addNode("decide", ["analyze"], 2, async ({ analyze }) => {
      if (!analyze) return "ABORT";
      // Lógica de Limiar
      if (analyze.balance < 0.6) return "REBALANCE_REQUIRED";
      return "MAINTAIN";
    });

    // 4. Intervenção (Alto Custo)
    dag.addNode("act", ["decide"], 40, async ({ decide }) => {
      if (decide === "REBALANCE_REQUIRED") {
        console.log("   ⚠️  Desequilíbrio detectado. Iniciando reestruturação...");
        systemTree = TreeOps.rebalance(systemTree);
        return "REBALANCED";
      }
      return "NO_ACTION";
    });

    // Execução do Grafo
    const result = await dag.run();
    currentCapital = result.capital + 20; // Injeção de capital por ciclo (Input Externo)

    // Log de Estado
    const metrics = result.results.get("analyze");
    const action = result.results.get("act");
    if (metrics) {
      console.log(`   [STATUS] Balance (Smooth): ${metrics.balance.toFixed(3)} | Ação: ${action || "N/A"}`);
    }
  }

  console.log("\n>>> SISTEMA ENCERRADO EM ESTADO ESTÁVEL <<<");
}

// Bootstrap
if (require.main === module) {
  runSimulation().catch(console.error);
}

module.exports = {
  TreeNode,
  TreeMetrics,
  TreeOps,
  LinearSignalRestorer,
  EconomicDAG,
  runSimulation
};
