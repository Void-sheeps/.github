/**
 * SCRIPT DOC — CAMPO SEMÂNTICO (SEMANTIC FIELD)
 * --------------------------------------------
 *
 * Estrutura de representação semântica baseada em operadores estruturais,
 * intensificação logística e projeções recursivas fundamentadas na razão áurea (φ).
 *
 * Componentes:
 * - SemanticNode: Unidade mínima de significado com valor dinâmico.
 * - Operators: Conjunto de transformações (log, sqrt, phi, square).
 * - SemanticField: Grafo/Conjunto de nós com capacidades de projeção estrutural.
 */

const PHI = (1 + Math.sqrt(5)) / 2;

/**
 * Função logística para saturação de valores.
 * @param {number} z - Valor de entrada.
 * @returns {number} Valor saturado entre 0 e 1.
 */
function logistic(z) {
  return 1 / (1 + Math.exp(-z));
}

// Operadores estruturais
const Operators = {
  log: (x) => Math.log10(Math.abs(x) + 1),
  sqrt: (x) => Math.sqrt(Math.abs(x)),
  phi: (x) => x * PHI,
  square: (x) => x * x
};

/**
 * Node Semântico
 */
class SemanticNode {
  constructor(label) {
    this.label = label;
    this.value = NaN;
  }

  /**
   * Aplica um operador estrutural ao nó.
   * @param {string} operatorName
   * @param {number} input
   */
  applyOperator(operatorName, input) {
    const op = Operators[operatorName];
    if (!op) throw new Error("Operador inexistente");

    this.value = op(input);
  }

  /**
   * Intensifica o valor do nó através de repetição.
   * @param {number} times - Número de repetições.
   */
  intensify(times = 1) {
    let z = isNaN(this.value) ? 0 : this.value;
    for (let i = 0; i < times; i++) {
      z += 1;
    }
    this.value = logistic(z);
  }
}

/**
 * Campo Semântico
 */
class SemanticField {
  constructor() {
    this.nodes = {};
  }

  /**
   * Adiciona uma nova palavra ao campo.
   * @param {string} word
   */
  addWord(word) {
    if (!this.nodes[word]) {
        this.nodes[word] = new SemanticNode(word);
    }
  }

  /**
   * Transforma uma palavra usando um operador.
   * @param {string} word
   * @param {string} operator
   * @param {number} input
   */
  transform(word, operator, input) {
    if (!this.nodes[word]) this.addWord(word);
    this.nodes[word].applyOperator(operator, input);
  }

  /**
   * Repete uma palavra para intensificar seu valor.
   * @param {string} word
   * @param {number} times
   */
  repeat(word, times) {
    if (!this.nodes[word]) this.addWord(word);
    this.nodes[word].intensify(times);
  }

  /**
   * Projeta uma estrutura recursiva baseada em Fibonacci/Phi.
   * @param {number} sequenceLength
   * @returns {Array} Sequência projetada.
   */
  projectPhiStructure(sequenceLength = 10) {
    let sequence = [1, 1];
    for (let i = 2; i < sequenceLength; i++) {
      sequence[i] = sequence[i - 1] + sequence[i - 2];
    }

    return sequence.map((v, i) => ({
      index: i,
      raw: v,
      phiRatio: v / (sequence[i - 1] || 1)
    }));
  }

  /**
   * Retorna o estado atual do campo.
   * @returns {Array}
   */
  state() {
    return Object.entries(this.nodes).map(([k, v]) => ({
      word: k,
      value: v.value
    }));
  }
}

if (require.main === module) {
    console.log("--- Semantic Field Validation ---");

    const field = new SemanticField();

    // Teste 1: Adição e Transformação
    field.addWord("tempo");
    field.transform("tempo", "log", 99); // log10(99+1) = 2
    const state1 = field.state().find(n => n.word === "tempo");
    console.log(`Test 1 (Transform): tempo value = ${state1.value}`);
    if (Math.abs(state1.value - 2) > 0.0001) {
        console.error("X Test 1 Failed");
        process.exit(1);
    }

    // Teste 2: Intensificação
    field.repeat("tempo", 1); // z = 2 + 1 = 3, logistic(3) approx 0.9525
    const state2 = field.state().find(n => n.word === "tempo");
    console.log(`Test 2 (Intensify): tempo value = ${state2.value}`);
    if (state2.value <= 0.95 || state2.value >= 0.96) {
        // precise: 1 / (1 + exp(-3)) = 0.95257
        console.error(`X Test 2 Failed: expected ~0.9525, got ${state2.value}`);
        process.exit(1);
    }

    // Teste 3: Estrutura Phi
    const phiStructure = field.projectPhiStructure(5);
    console.log("Test 3 (Phi Structure):", phiStructure[4]);
    if (phiStructure[4].raw !== 5) { // 1, 1, 2, 3, 5
        console.error("X Test 3 Failed");
        process.exit(1);
    }

    console.log("✓ All semantic field validations passed.");
}

module.exports = {
    SemanticNode,
    SemanticField,
    Operators,
    logistic
};
