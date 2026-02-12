const EventEmitter = require('events');

/* ============================================================
 *  LinearSignalRestorer — filtro com EMA, extrapolação e confiança
 * ============================================================ */

class LinearSignalRestorer {
  constructor({
    constraints = { min: 0, max: 100, maxDerivative: 20 },
    reconstructionWeight = 0.7,
    predictionWeight = 0.3,
    deviationPower = 2,
    deadband = 0.0001,
    emaAlpha = 0.4,
    enableMultiStep = true,
    maxExtrapolationSteps = 5,
    trendDecay = 0.92,
    confidenceRecoveryRate = 0.05,
    confidenceDecayRate = 0.95,
  } = {}) {
    this.constraints = constraints;
    this.reconstructionWeight = reconstructionWeight;
    this.predictionWeight = predictionWeight;
    this.deviationPower = deviationPower;
    this.deadband = deadband;
    this.emaAlpha = emaAlpha;
    this.enableMultiStep = enableMultiStep;
    this.maxExtrapolationSteps = maxExtrapolationSteps;
    this.trendDecay = trendDecay;
    this.confidenceRecoveryRate = confidenceRecoveryRate;
    this.confidenceDecayRate = confidenceDecayRate;

    this.lastValidState = 50;
    this.trend = 0;
    this.outageCounter = 0;
    this.confidence = 1.0;
  }

  _isFinite(x) {
    return typeof x === 'number' && Number.isFinite(x);
  }

  _clamp(x) {
    const { min, max } = this.constraints;
    return Math.max(min, Math.min(max, x));
  }

  _clampDerivative(x, last) {
    const delta = x - last;
    const maxD = this.constraints.maxDerivative;
    if (Math.abs(delta) > maxD) {
      return last + Math.sign(delta) * maxD;
    }
    return x;
  }

  _applyDeadband(x, last) {
    if (Math.abs(x - last) < this.deadband) {
      return last;
    }
    return x;
  }

  _weight(deviation) {
    return 1 / (1 + Math.pow(Math.abs(deviation), this.deviationPower));
  }

  _updateTrend(newVal, isOutage = false) {
    const delta = newVal - this.lastValidState;
    const alpha = isOutage ? this.emaAlpha * 0.2 : this.emaAlpha;

    this.trend = alpha * delta + (1 - alpha) * this.trend;

    if (isOutage) {
      this.outageCounter++;
      this.confidence *= this.confidenceDecayRate;
    } else {
      this.outageCounter = 0;
      this.confidence = Math.min(1, this.confidence + this.confidenceRecoveryRate);
    }
  }

  _predict() {
    if (!this.enableMultiStep || this.outageCounter <= 1) {
      return this.lastValidState + this.trend;
    }

    const steps = Math.min(this.outageCounter, this.maxExtrapolationSteps);
    const decayFactor = Math.pow(this.trendDecay, steps - 1);

    return this.lastValidState + this.trend * steps * decayFactor * this.confidence;
  }

  reconstruct(primary, secondaries = []) {
    const last = this.lastValidState;
    const predicted = this._predict();

    if (this._isFinite(primary)) {
      let val = this._clamp(primary);
      val = this._applyDeadband(val, last);
      val = this._clampDerivative(val, last);

      this._updateTrend(val, false);
      this.lastValidState = val;

      return {
        value: val,
        method: 'DIRECT',
        status: 'OK',
        confidence: this.confidence,
        outageSteps: this.outageCounter,
      };
    }

    const validSecondaries = (secondaries || []).filter(v => this._isFinite(v));

    if (validSecondaries.length === 0) {
      let fallback = this._clampDerivative(predicted, last);
      fallback = this._clamp(fallback);
      fallback = this._applyDeadband(fallback, last);

      this._updateTrend(fallback, true);
      this.lastValidState = fallback;

      return {
        value: parseFloat(fallback.toFixed(4)),
        method: 'TREND_FALLBACK',
        status: 'DEGRADED',
        confidence: this.confidence,
        outageSteps: this.outageCounter,
      };
    }

    const weights = validSecondaries.map(v => this._weight(v - predicted));
    const weightSum = weights.reduce((a, b) => a + b, 0);

    const weightedSum = validSecondaries.reduce(
      (acc, v, i) => acc + v * (weights[i] / weightSum),
      0
    );

    let result = weightedSum * this.reconstructionWeight + predicted * this.predictionWeight;

    result = this._clampDerivative(result, last);
    result = this._clamp(result);
    result = this._applyDeadband(result, last);

    this._updateTrend(result, false);
    this.lastValidState = result;

    return {
      value: parseFloat(result.toFixed(4)),
      method: 'ADAPTIVE',
      status: 'RECOVERED',
      confidence: this.confidence,
      outageSteps: this.outageCounter,
    };
  }

  apply(primary, secondaries = []) {
    return this.reconstruct(primary, secondaries);
  }
}

/* ==========================================================================
   TopologicalProcessor — executor DAG resiliente com políticas de erro
   ========================================================================== */

class TopologicalProcessor extends EventEmitter {
  constructor({
    concurrency = 4,
    errorPolicy = 'propagate', // 'propagate' | 'isolate' | 'ignore'
    defaultTimeoutMs = 15000,
    globalTimeoutMs = null,
  } = {}) {
    super();
    this.concurrency = concurrency;
    this.errorPolicy = errorPolicy;
    this.defaultTimeoutMs = defaultTimeoutMs;
    this.globalTimeoutMs = globalTimeoutMs;

    this.graph = new Map();
    this.inDegree = new Map();
    this.queue = [];
    this.running = 0;
    this.results = new Map();
    this.abortController = new AbortController();
    this._draining = false;
  }

  addNode({
    id,
    fn,
    dependencies = [],
    timeout = null,
    canFail = false,
    fallbackValue = null,
  }) {
    if (this.graph.has(id)) {
      throw new Error(`Node ${id} already exists`);
    }

    this.graph.set(id, {
      id,
      fn,
      dependencies,
      dependents: new Set(),
      timeout: timeout ?? this.defaultTimeoutMs,
      canFail,
      fallbackValue,
      solved: false,
      failed: false,
    });

    this.inDegree.set(id, dependencies.length);

    dependencies.forEach(dep => {
      if (!this.graph.has(dep)) {
        throw new Error(`Dependency ${dep} not found for node ${id}`);
      }
      this.graph.get(dep).dependents.add(id);
    });

    if (dependencies.length === 0) {
      this.queue.push(id);
    }
  }

  async _executeNode(id) {
    const node = this.graph.get(id);
    if (!node || node.solved || node.failed) return;

    this.running++;

    try {
      const context = Object.fromEntries(
        node.dependencies.map(dep => [dep, this.results.get(dep)])
      );

      const signal = this.abortController.signal;
      signal.throwIfAborted();

      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout (${node.timeout}ms)`)), node.timeout)
      );

      const result = await Promise.race([
        node.fn(context, signal),
        timeoutPromise,
      ]);

      this.results.set(id, result);
    } catch (err) {
      if (this.abortController.signal.aborted) {
        node.failed = true;
        this.running--;
        return;
      }

      if (node.canFail) {
        this.results.set(id, node.fallbackValue);
      } else {
        node.failed = true;

        if (this.errorPolicy === 'propagate') {
          this.abortController.abort();
          this.emit('error', { id, error: err });
          this.running--;
          return;
        } else if (this.errorPolicy === 'isolate') {
          this._failDownstream(id);
        }
      }
    }

    node.solved = true;
    this.running--;

    for (const depId of node.dependents) {
      const newDegree = this.inDegree.get(depId) - 1;
      this.inDegree.set(depId, newDegree);
      if (newDegree === 0) {
        this.queue.push(depId);
      }
    }

    this._drain();
  }

  _failDownstream(id) {
    const node = this.graph.get(id);
    if (!node) return;

    for (const depId of node.dependents) {
      const depNode = this.graph.get(depId);
      if (depNode && !depNode.failed) {
        depNode.failed = true;
        this._failDownstream(depId);
      }
    }
  }

  _drain() {
    if (this._draining) return;
    this._draining = true;

    while (this.running < this.concurrency && this.queue.length > 0) {
      const id = this.queue.shift();
      this._executeNode(id).catch(err => {
        this.emit('internalError', err);
      });
    }

    if (this.running === 0 && this.queue.length === 0) {
      const unresolved = [...this.graph.values()].filter(n => !n.solved && !n.failed);
      if (unresolved.length > 0) {
        this.emit('stalled', { unresolved: unresolved.map(n => n.id) });
      } else {
        this.emit('drain', { results: this.results });
      }
    }

    this._draining = false;
  }

  async run() {
    if (this.globalTimeoutMs) {
      setTimeout(() => this.abortController.abort(), this.globalTimeoutMs);
    }

    return new Promise((resolve, reject) => {
      const onDrain = (data) => {
        this.off('drain', onDrain);
        this.off('error', onError);
        resolve(data.results);
      };

      const onError = (info) => {
        this.off('drain', onDrain);
        this.off('error', onError);
        reject(info);
      };

      this.once('drain', onDrain);
      this.once('error', onError);

      this._drain();
    });
  }

  cancel() {
    this.abortController.abort();
  }

  clear() {
    this.graph.clear();
    this.inDegree.clear();
    this.results.clear();
    this.queue.length = 0;
    this.running = 0;
    this.abortController = new AbortController();
  }
}

// Bloco de Autovalidação
if (require.main === module) {
  (async () => {
    console.log('--- Resilient Topo System (Superior Version) Validation ---');
    try {
      const restorer = new LinearSignalRestorer();
      console.log('1. Testing Adaptive Reconstruction...');
      restorer.apply(50);
      const res = restorer.apply(null, [52, 53]);
      if (res.method !== 'ADAPTIVE') throw new Error(`Expected ADAPTIVE, got ${res.method}`);
      console.log(`   Result: ${res.value}, Method: ${res.method}`);

      console.log('2. Testing Topological DAG with Async execution...');
      const tp = new TopologicalProcessor({ concurrency: 2 });
      tp.addNode({ id: 'A', fn: async () => 'ResultA' });
      tp.addNode({ id: 'B', fn: async (ctx) => ctx.A + '->ResultB', dependencies: ['A'] });

      const results = await tp.run();
      if (results.get('B') === 'ResultA->ResultB') {
        console.log('✓ DAG executed successfully.');
      } else {
        throw new Error('DAG failed context propagation.');
      }

      console.log('3. Testing Error Policy (isolate)...');
      const tp2 = new TopologicalProcessor({ concurrency: 1, errorPolicy: 'isolate' });
      tp2.addNode({ id: 'Root', fn: async () => { throw new Error('Root failed'); } });
      tp2.addNode({ id: 'Dependent', fn: async () => 'Should not run', dependencies: ['Root'] });

      const res2 = await tp2.run();
      const depNode = tp2.graph.get('Dependent');
      if (depNode.failed && !res2.has('Dependent')) {
        console.log('✓ Isolation verified: dependent marked as failed and excluded from results.');
      } else {
        throw new Error('Isolation failed: dependent was not correctly handled.');
      }

      console.log('All validations passed successfully.');
    } catch (err) {
      console.error('Validation FAILED:', err);
      process.exit(1);
    }
  })();
}

module.exports = {
  LinearSignalRestorer,
  TopologicalProcessor
};
