// mapa_ontologico_refinado.js
const NeighborhoodError = {
  EmptyNeighborhood: 'ERRO: Vizinhança Vazia',
  NonVectorConcept: (idx) => `ERRO: Conceito Não-Vetorial no Índice ${idx}`,
  DimensionMismatch: (expected, received) =>
    `ERRO: Descompasso de Dimensão (Esperado: ${expected}, Recebido: ${received})`,
  InvalidMatrix: 'ERRO: Matriz de contexto inválida'
};

class MapaOntologico {
  constructor() {
    this.resolutions = {
      '8bit': Uint8Array,
      '16bit': Uint16Array,
      '32bit': Uint32Array,
      '64bit': Float64Array
    };
  }

  _isTypedArray(x) {
    return ArrayBuffer.isView(x) && !(x instanceof DataView);
  }

  _inferOutputType(vector) {
    if (vector instanceof Float64Array) return Float64Array;
    if (vector instanceof Float32Array) return Float64Array;
    if (vector instanceof Uint32Array) return Float64Array;
    if (vector instanceof Uint16Array) return Uint16Array;
    if (vector instanceof Uint8Array) return Uint8Array;
    if (vector instanceof BigInt64Array) return BigInt64Array;
    return Float64Array;
  }

  _validateMatrix(matrix, vSize) {
    if (!Array.isArray(matrix) || matrix.length === 0) {
      throw new RangeError(NeighborhoodError.EmptyNeighborhood);
    }
    for (let r = 0; r < matrix.length; r++) {
      const row = matrix[r];
      if (!Array.isArray(row)) throw new TypeError(NeighborhoodError.NonVectorConcept(r));
      if (row.length !== vSize) throw new RangeError(NeighborhoodError.DimensionMismatch(vSize, row.length));
      for (let c = 0; c < row.length; c++) {
        const v = row[c];
        if (typeof v !== 'number' && typeof v !== 'bigint') throw new TypeError(NeighborhoodError.InvalidMatrix);
      }
    }
  }

  applyContext(vector, matrix) {
    if (!this._isTypedArray(vector)) throw new TypeError('vector must be a TypedArray');
    const vSize = vector.length;
    this._validateMatrix(matrix, vSize);

    const OutType = this._inferOutputType(vector);
    const out = (OutType === BigInt64Array) ? new BigInt64Array(matrix.length) : new OutType(matrix.length);

    // Use loops for performance and numeric stability
    for (let r = 0; r < matrix.length; r++) {
      const row = matrix[r];
      if (OutType === BigInt64Array) {
        let acc = 0n;
        for (let i = 0; i < vSize; i++) {
          acc += BigInt(row[i]) * BigInt(vector[i]);
        }
        out[r] = acc;
      } else {
        let acc = 0.0;
        for (let i = 0; i < vSize; i++) {
          acc += row[i] * Number(vector[i]);
        }
        out[r] = acc;
      }
    }

    return out;
  }
}

if (require.main === module) {
  // Simple validation run if executed directly
  try {
    const mo = new MapaOntologico();
    const vec = new Uint8Array([1, 2, 3]);
    const mat = [[10, 20, 30], [5, 10, 15]];
    const res = mo.applyContext(vec, mat);
    console.log('Validation successful. Result:', res);
  } catch (e) {
    console.error('Validation failed:', e.message);
    process.exit(1);
  }
}

module.exports = { MapaOntologico, NeighborhoodError };
