/**
 * SCRIPT DOC — MAPA ONTOLÓGICO SIMBÓLICO
 * ---------------------------------------
 *
 * Estrutura Formal:
 * 𝓜 = (𝓢, 𝓞, 𝓒, Q, Φ, A)
 *
 * 𝓢 = Conjunto de Signos
 * 𝓞 = Conjunto de Operadores
 * 𝓒 = Conjunto de Contextos
 *
 * Q  : 𝓢 → 𝕂                (Estado Interno / Quanta)
 * Φ  : 𝓒 → 𝓞                (Contextualização)
 * A  : 𝓢 × T → 𝕂            (Modulação Axial)
 *
 * Aplicação Geral:
 *      s' = Φ(c)(s)
 *
 * Dinâmica Temporal:
 *      s(t) = A(s, t)
 *
 * Transformação Linear (caso vetorial):
 *      v' = Mv
 *
 * Estado Categorial:
 *      Σ(s) = (C(s), G(s))
 *
 *      onde:
 *          C(s) ∈ {0,1}     (Natural / Transcendental)
 *          G(s) = (Q(s) - θ) mod Λ
 */

const OntoConst = require('./onto_const');

/**
 * @typedef {Object} Signo
 * @property {*} id                Identificador do signo
 * @property {number|bigint|any} quanta   Estado interno Q(s)
 */


/**
 * @typedef {function(Signo): Signo} Operador
 * Operador F ∈ 𝓞
 * F : 𝓢 → 𝓢
 */


/**
 * @typedef {function(any): Operador} Contexto
 * Φ : 𝓒 → 𝓞
 * Um contexto seleciona ou constrói um operador.
 */


/**
 * @typedef {function(Signo, number): number} Axial
 * A : 𝓢 × T → 𝕂
 * Modulação temporal ou paramétrica.
 */


/**
 * @function aplicarContexto
 * @description
 * Aplica contextualização ontológica.
 *
 * Formal:
 *      s' = Φ(c)(s)
 *
 * @param {Signo} s
 * @param {Contexto} Φ
 * @param {*} c
 * @returns {Signo}
 */
function aplicarContexto(s, Φ, c) {
    const F = Φ(c);
    return F(s);
}


/**
 * @function modularAxial
 * @description
 * Modulação axial temporal.
 *
 * Formal:
 *      A(s, t) = Q(s) · ω(t)
 *
 * @param {Signo} s
 * @param {function(number): number} ω
 * @param {number} t
 * @returns {number}
 */
function modularAxial(s, ω, t) {
    return s.quanta * ω(t);
}


/**
 * @function estadoCategorial
 * @description
 * Avalia estado categorial segundo limiar θ e limite Λ.
 *
 * Formal:
 *      C(s) =
 *          0 se Q(s) ≥ θ
 *          1 se Q(s) < θ e evolui
 *          overflow se Q(s) ≥ Λ
 *
 *      G(s) = (Q(s) - θ) mod Λ
 *
 * @param {Signo} s
 * @param {number} θ  Limiar estrutural
 * @param {number} Λ  Limite estrutural máximo
 * @returns {{categoria: string, grau: number}}
 */
function estadoCategorial(s, θ = OntoConst.LIMIAR, Λ = OntoConst.LIMITE_MAXIMO) {
    const Q = s.quanta;

    if (Q >= Λ) {
        return { categoria: OntoConst.CATEGORIA.OVERFLOW, grau: 0 };
    }

    if (Q >= θ) {
        return {
            categoria: OntoConst.CATEGORIA.NATURAL,
            grau: (Q - θ) % Λ
        };
    }

    return {
        categoria: OntoConst.CATEGORIA.TRANSCENDENTAL,
        grau: (Q - θ + Λ) % Λ
    };
}

if (require.main === module) {
    console.log("--- Symbolic Ontological Map Validation ---");

    const θ = OntoConst.LIMIAR;
    const Λ = OntoConst.LIMITE_MAXIMO;

    const s1 = { id: "S1", quanta: 20 };
    const res1 = estadoCategorial(s1, θ, Λ);
    console.log(`Test 1 (Natural): Q=${s1.quanta}, θ=${θ} -> ${res1.categoria}, Grau: ${res1.grau}`);
    if (res1.categoria !== OntoConst.CATEGORIA.NATURAL || res1.grau !== 5) {
        console.error("X Test 1 Failed");
        process.exit(1);
    }

    const s2 = { id: "S2", quanta: 10 };
    const res2 = estadoCategorial(s2, θ, Λ);
    console.log(`Test 2 (Transcendental): Q=${s2.quanta}, θ=${θ} -> ${res2.categoria}, Grau: ${res2.grau}`);
    if (res2.categoria !== OntoConst.CATEGORIA.TRANSCENDENTAL || res2.grau !== 122) { // 10-15+127 = 122
        console.error(`X Test 2 Failed: expected 122, got ${res2.grau}`);
        process.exit(1);
    }

    const s3 = { id: "S3", quanta: 127 };
    const res3 = estadoCategorial(s3, θ, Λ);
    console.log(`Test 3 (Overflow): Q=${s3.quanta}, Λ=${Λ} -> ${res3.categoria}`);
    if (res3.categoria !== OntoConst.CATEGORIA.OVERFLOW) {
        console.error("X Test 3 Failed");
        process.exit(1);
    }

    // Test modularAxial
    const ω = OntoConst.MODULACAO.senoidal;
    const val = modularAxial(s1, ω, Math.PI / 2);
    console.log(`Test 4 (Axial): Q=${s1.quanta}, sin(π/2) -> ${val}`);
    if (Math.abs(val - 20) > 0.0001) {
        console.error("X Test 4 Failed");
        process.exit(1);
    }

    console.log("✓ All symbolic map validations passed.");
}

module.exports = {
    aplicarContexto,
    modularAxial,
    estadoCategorial
};
