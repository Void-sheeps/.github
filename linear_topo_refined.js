const EventEmitter = require('events');

class LinearSignalRestorer {
  constructor({ constraints = {min:0, max:100, maxDerivative:20}, adaptWeights=false } = {}) {
    this.constraints = constraints;
    this.adaptWeights = adaptWeights;
    this.lastValidState = 50; // default
  }

  _isFiniteNumber(x) {
    return typeof x === 'number' && Number.isFinite(x);
  }

  isStatisticallyValid(measurement, lastState = this.lastValidState) {
    if (!this._isFiniteNumber(measurement)) return false;
    if (measurement < this.constraints.min || measurement > this.constraints.max) return false;
    if (Math.abs(measurement - lastState) > this.constraints.maxDerivative) return false;
    return true;
  }

  /**
   * Pure function: retorna { value, method, status, newState } sem mutar this.
   * Opcional: pass lastState para reentrância.
   */
  reconstruct(rawPrimary, secondarySensors = [], { lastState = this.lastValidState, weights = null } = {}) {
    if (this.isStatisticallyValid(rawPrimary, lastState)) {
      return { value: rawPrimary, method: 'DIRECT_READ', status: 'OK', newState: rawPrimary };
    }

    // valida secundários
    const validSecondaries = (secondarySensors || []).filter(s => this._isFiniteNumber(s));
    if (validSecondaries.length === 0) {
      // fallback: retorna lastState (inércia) com sinal de degradação
      return { value: lastState, method: 'FALLBACK_INERTIA', status: 'DEGRADED', newState: lastState };
    }

    // pesos: se fornecidos, normaliza; senão, usa heurística simples
    let wArr;
    if (Array.isArray(weights) && weights.length === validSecondaries.length + 1) {
      // weights: [w_sec1, ..., w_secn, w_time]
      const sum = weights.reduce((s,v)=>s+v,0);
      wArr = weights.map(w => w / sum);
    } else {
      // heurística: média dos secundários com peso temporal menor
      const wTime = 0.3;
      const wSecTotal = 1 - wTime;
      const perSec = wSecTotal / validSecondaries.length;
      wArr = [...Array(validSecondaries.length).fill(perSec), wTime];
    }

    // reconstrução linear: sum(w_i * sec_i) + w_time * lastState
    let acc = 0;
    for (let i = 0; i < validSecondaries.length; i++) {
      acc += wArr[i] * validSecondaries[i];
    }
    acc += wArr[wArr.length - 1] * lastState;

    // sanitize numeric result
    if (!Number.isFinite(acc)) acc = lastState;

    return { value: parseFloat(acc.toFixed(4)), method: 'LINEAR_RECONSTRUCTION', status: 'RECOVERED', newState: acc };
  }

  // mutating convenience wrapper
  apply(rawPrimary, secondarySensors = [], opts = {}) {
    const res = this.reconstruct(rawPrimary, secondarySensors, opts);
    this.lastValidState = res.newState;
    return res;
  }
}

/* ---------------- TopologicalProcessor refinado ---------------- */

class TopologicalProcessor extends EventEmitter {
  constructor({ concurrency = 2, nodeTimeout = 10000 } = {}) {
    super();
    this.graph = new Map();
    this.inDegree = new Map();
    this.queue = [];
    this.concurrency = concurrency;
    this.running = 0;
    this.nodeTimeout = nodeTimeout;
  }

  addNode(id, rawData, dependencies = []) {
    if (!this.graph.has(id)) {
      this.graph.set(id, { id, data: rawData, dependents: new Set(), solved: false });
      this.inDegree.set(id, 0);
    } else {
      this.graph.get(id).data = rawData;
    }

    // ensure dependency nodes exist and build edges
    dependencies.forEach(depId => {
      if (!this.graph.has(depId)) {
        this.graph.set(depId, { id: depId, data: null, dependents: new Set(), solved: false });
        this.inDegree.set(depId, 0);
      }
      // add edge depId -> id
      this.graph.get(depId).dependents.add(id);
      this.inDegree.set(id, (this.inDegree.get(id) || 0) + 1);
    });

    // if in-degree zero, enqueue
    if ((this.inDegree.get(id) || 0) === 0) this._enqueue(id);
  }

  _enqueue(id) {
    if (!this.queue.includes(id)) this.queue.push(id);
    // try to process
    this._drain();
  }

  _drain() {
    while (this.running < this.concurrency && this.queue.length > 0) {
      const id = this.queue.shift();
      this._processNode(id);
    }
  }

  _processNode(id) {
    const node = this.graph.get(id);
    if (!node || node.solved) return;
    this.running++;
    node.processing = true;

    const isNoise = (node.data === null);
    const processingTime = isNoise ? 3000 : 100;

    // support timeout and cancellation
    const timer = setTimeout(() => {
      // simulate result or timeout fallback
      const result = isNoise ? 'RECONSTRUCTED_VIA_MATRIX' : node.data;
      this._completeNode(id, result);
    }, processingTime);

    // also enforce nodeTimeout
    const to = setTimeout(() => {
      if (!node.solved) {
        clearTimeout(timer);
        this._completeNode(id, 'TIMEOUT_FALLBACK');
      }
    }, this.nodeTimeout);

    // store timers for potential cancellation
    node._timers = { timer, to };
  }

  _completeNode(id, result) {
    const node = this.graph.get(id);
    if (!node || node.solved) return;
    node.solved = true;
    node.processing = false;
    this.running--;
    // clear timers
    if (node._timers) {
      clearTimeout(node._timers.timer);
      clearTimeout(node._timers.to);
      delete node._timers;
    }

    this.emit('solved', { id, result });
    // decrement in-degree of dependents
    for (const depId of node.dependents) {
      const cur = (this.inDegree.get(depId) || 1) - 1;
      this.inDegree.set(depId, cur);
      if (cur === 0) this._enqueue(depId);
    }

    // detect completion/cycle: if no running and queue empty, check for unsolved nodes
    if (this.running === 0 && this.queue.length === 0) {
      const unsolved = Array.from(this.graph.values()).filter(n => !n.solved);
      if (unsolved.length > 0) {
        // cycle or missing deps
        this.emit('stalled', { unsolved: unsolved.map(n => n.id) });
      } else {
        this.emit('drain');
      }
    }

    // continue draining
    this._drain();
  }

  // convenience: wait until all processed or stalled
  async waitAll(timeout = 30000) {
    return new Promise((resolve, reject) => {
      const onDrain = () => cleanup(resolve, null);
      const onStalled = (info) => cleanup(null, info);
      const to = setTimeout(() => cleanup(null, { reason: 'global_timeout' }), timeout);

      const cleanup = (ok, err) => {
        clearTimeout(to);
        this.off('drain', onDrain);
        this.off('stalled', onStalled);
        if (err) reject(err); else resolve(ok);
      };

      this.on('drain', onDrain);
      this.on('stalled', onStalled);
      // kick drain in case queue already has items
      this._drain();
    });
  }
}

if (require.main === module) {
  (async () => {
    try {
      console.log('Starting validation for LinearSignalRestorer...');
      const restorer = new LinearSignalRestorer();
      const res1 = restorer.reconstruct(55);
      if (res1.value !== 55 || res1.method !== 'DIRECT_READ') throw new Error('Restorer DIRECT_READ failed');

      const res2 = restorer.reconstruct(150); // out of bounds [0, 100]
      if (res2.method !== 'FALLBACK_INERTIA') throw new Error('Restorer FALLBACK_INERTIA failed');

      console.log('LinearSignalRestorer validation passed.');

      console.log('Starting validation for TopologicalProcessor...');
      const tp = new TopologicalProcessor({ concurrency: 2 });
      tp.addNode('A', 'dataA');
      tp.addNode('B', 'dataB', ['A']);

      const results = [];
      tp.on('solved', ({ id, result }) => {
        results.push({ id, result });
      });

      await tp.waitAll(2000);
      if (results.length !== 2) throw new Error(`Expected 2 solved nodes, got ${results.length}`);
      if (results[0].id !== 'A' || results[1].id !== 'B') throw new Error('Topological order failed');

      console.log('TopologicalProcessor validation passed.');
      console.log('All validations successful.');
    } catch (e) {
      console.error('Validation failed:', e.message);
      process.exit(1);
    }
  })();
}

module.exports = { LinearSignalRestorer, TopologicalProcessor };
