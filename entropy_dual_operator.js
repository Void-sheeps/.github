/**
 * ===============================================================
 *  ENTROPY DUAL OPERATOR (Φ)
 * ===============================================================
 *
 * Formalização:
 *
 * Seja:
 *   v(x,y)  = valor local do campo
 *   μ(x,y)  = média local (estrutura)
 *   δ(x,y)  = v - μ (divergência)
 *
 * Operador geométrico da dualidade:
 *
 *   Φ(x,y) = √( μ² + δ² )
 *
 * Interpretação:
 *   μ  → componente convergente (estrutura)
 *   δ  → componente divergente (polarização)
 *   Φ  → magnitude total da tensão estrutural
 *
 * Propriedades:
 *   - Φ ≥ 0
 *   - Cresce tanto com ordem quanto com ruptura
 *   - Não cancela polaridade
 *
 * Este módulo:
 *   1. Gera heatmap espacial Φ
 *   2. Calcula entropia de Shannon real do campo
 *   3. Ajusta diffusionRate dinamicamente
 *   4. Implementa modo auto-stabilizing
 *
 * ===============================================================
 */

class EntropyDualOperator {

  constructor(field, options = {}) {
    this.field = field;

    this.size = field.size;
    this.area = field.area;

    this.map = new Float32Array(this.area);

    // Controle adaptativo
    this.baseDiffusion = field.diffusionRate;
    this.adaptStrength = options.adaptStrength ?? 0.5;
    this.entropyTarget = options.entropyTarget ?? 0.6;
    this.autoStabilize = options.autoStabilize ?? true;
  }

  _idx(x, y) {
    const s = this.size;
    const wx = (x + s) % s;
    const wy = (y + s) % s;
    return wy * s + wx;
  }

  _localMean(x, y) {
    const f = this.field.current;

    const nN = f[this._idx(x, y - 1)];
    const nS = f[this._idx(x, y + 1)];
    const nE = f[this._idx(x + 1, y)];
    const nW = f[this._idx(x - 1, y)];

    return (nN + nS + nE + nW) * 0.25;
  }

  /**
   * ===============================================================
   * 1. HEATMAP Φ
   * ===============================================================
   */
  computePhiMap() {
    const f = this.field.current;
    const s = this.size;

    let totalPhi = 0;

    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const i = y * s + x;

        const v = f[i];
        const mu = this._localMean(x, y);
        const delta = v - mu;

        const phi = Math.sqrt(mu * mu + delta * delta);

        this.map[i] = phi;
        totalPhi += phi;
      }
    }

    return {
      heatmap: this.map,
      totalPhi,
      meanPhi: totalPhi / this.area
    };
  }

  /**
   * ===============================================================
   * 2. ENTROPIA DE SHANNON REAL
   * ===============================================================
   *
   * H = - Σ p_i log2(p_i)
   *
   * p_i = v_i / Σ v
   */
  computeShannonEntropy() {
    const data = this.field.current;
    const len = data.length;

    let total = 0;
    for (let i = 0; i < len; i++) {
      total += data[i];
    }

    if (total === 0) return 0;

    let entropy = 0;

    for (let i = 0; i < len; i++) {
      const v = data[i];
      if (v > 0) {
        const p = v / total;
        entropy -= p * Math.log2(p);
      }
    }

    // Normalização (0–1)
    const maxEntropy = Math.log2(len);
    return entropy / maxEntropy;
  }

  /**
   * ===============================================================
   * 3. FEEDBACK ADAPTATIVO
   * ===============================================================
   *
   * Se entropia > alvo → reduz difusão (evita colapso)
   * Se entropia < alvo → aumenta difusão (evita cristalização)
   */
  adaptDiffusion(entropy) {
    if (!this.autoStabilize) return;

    const deviation = entropy - this.entropyTarget;

    const adjustment = 1 - (deviation * this.adaptStrength);

    const newRate = this.baseDiffusion * adjustment;

    // Clamp físico
    this.field.diffusionRate = Math.max(0.001, Math.min(1.0, newRate));
  }

  /**
   * ===============================================================
   * 4. CICLO COMPLETO AUTO-STABILIZANTE
   * ===============================================================
   */
  step() {
    const phiStats = this.computePhiMap();
    const entropy = this.computeShannonEntropy();

    this.adaptDiffusion(entropy);

    return {
      ...phiStats,
      entropy,
      diffusionRate: this.field.diffusionRate
    };
  }

}

if (require.main === module) {
  const size = 10;
  const field = {
    size: size,
    area: size * size,
    current: new Float32Array(size * size).fill(0.5),
    diffusionRate: 0.1
  };

  // Add some "entropy"
  field.current[5] = 1.0;
  field.current[55] = 0.2;

  try {
    const edo = new EntropyDualOperator(field);
    const result = edo.step();

    console.log("EntropyDualOperator Validation:");
    console.log("- Mean Phi:", result.meanPhi.toFixed(4));
    console.log("- Entropy:", result.entropy.toFixed(4));
    console.log("- New Diffusion Rate:", result.diffusionRate.toFixed(4));

    if (result.entropy > 0 && result.meanPhi >= 0) {
      console.log("✓ Validation passed.");
    } else {
      console.error("X Validation failed: unexpected values.");
      process.exit(1);
    }
  } catch (err) {
    console.error("X Validation failed with error:", err);
    process.exit(1);
  }
}

module.exports = EntropyDualOperator;
