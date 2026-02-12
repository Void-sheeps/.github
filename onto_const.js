/**
 * SCRIPT DOC — MÓDULO DE CONSTANTES ONTOLÓGICAS
 * ---------------------------------------------
 *
 * Este módulo define as constantes estruturais universais do
 * sistema ontológico simbólico.
 *
 * Estrutura:
 *   θ  = Limiar estrutural
 *   Λ  = Limite estrutural máximo
 *   C  = Categorias simbólicas
 *   Ω  = Funções modulares padrão
 *
 * Formal:
 *   θ ∈ 𝕂
 *   Λ ∈ 𝕂
 *   C ⊂ {Natural, Transcendental, Overflow}
 */

'use strict';

/**
 * @namespace OntoConst
 */
const OntoConst = Object.freeze({

    /**
     * Limiar Estrutural (θ)
     * Determina o ponto de estabilização natural.
     *
     * Formal:
     *     θ ∈ 𝕂
     */
    LIMIAR: 0x0F, // 15


    /**
     * Limite Estrutural Máximo (Λ)
     * Determina saturação / overflow estrutural.
     *
     * Formal:
     *     Λ ∈ 𝕂
     */
    LIMITE_MAXIMO: 0x7F, // 127


    /**
     * Categorias Ontológicas
     *
     * C(s) ∈ {Natural, Transcendental, Overflow}
     */
    CATEGORIA: Object.freeze({
        NATURAL: 'Natural',
        TRANSCENDENTAL: 'Transcendental',
        OVERFLOW: 'Overflow'
    }),


    /**
     * Codificação Binária Categorial
     *
     * bit7 = 0 → Natural
     * bit7 = 1 → Transcendental
     */
    BIT_CATEGORIA: 0x80,


    /**
     * Máscara de Grau (7 bits inferiores)
     */
    MASCARA_GRAU: 0x7F,


    /**
     * Funções de Modulação Axial Padrão
     *
     * ω : T → ℝ
     */
    MODULACAO: Object.freeze({

        /**
         * Oscilação Senoidal
         * ω(t) = sin(t)
         */
        senoidal: (t) => Math.sin(t),

        /**
         * Oscilação Cossenoidal
         */
        cossenoidal: (t) => Math.cos(t),

        /**
         * Modulação Linear
         */
        linear: (t) => t,

        /**
         * Modulação Identidade
         */
        identidade: () => 1
    })

});

if (require.main === module) {
    console.log("--- Ontological Constants Validation ---");
    console.log(`LIMIAR: ${OntoConst.LIMIAR}`);
    console.log(`LIMITE_MAXIMO: ${OntoConst.LIMITE_MAXIMO}`);
    console.log(`CATEGORIA NATURAL: ${OntoConst.CATEGORIA.NATURAL}`);

    if (OntoConst.LIMIAR === 15 && OntoConst.LIMITE_MAXIMO === 127) {
        console.log("✓ Constants verified.");
    } else {
        console.error("X Constants verification failed.");
        process.exit(1);
    }
}

module.exports = OntoConst;
