// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { Graphs, Mesh, Parameters } from '../src/model';
import { forceArrow } from '../src/render';
import { Simulation } from '../src/simulation';
import { conjugateGradient, dot, newtonCG, type Objective, type Vector } from '../src/solver';

test('CG solves SPD matrix-free system and reports actual work', async () => {
  const result = await conjugateGradient(
    async (v) => Float32Array.of(2 * v[0], 5 * v[1]),
    Float32Array.of(4, -10),
    0,
  );
  assert.ok(Math.abs(result.direction[0] + 2) < 1e-5);
  assert.ok(Math.abs(result.direction[1] - 2) < 1e-5);
  assert.equal(result.negativeCurvature, false);
  assert.equal(result.iterations, 2);
});

test('CG detects negative curvature and produces descent, including after a good iteration', async () => {
  for (const diagonal of [
    [-2, -1],
    [1, -0.5],
  ]) {
    const g = Float32Array.of(1, 0.1);
    const result = await conjugateGradient(
      async (v) => Float32Array.of(diagonal[0] * v[0], diagonal[1] * v[1]),
      g,
      0,
    );
    assert.equal(result.negativeCurvature, true);
    assert.ok(dot(g, result.direction) < 0);
    assert.ok(result.iterations <= 2);
  }
});

test('zero gradient uses no CG work; non-finite HVP fails explicitly', async () => {
  let calls = 0;
  const result = await conjugateGradient(
    async (v) => {
      calls++;
      return v;
    },
    Float32Array.of(0, 0),
    0,
  );
  assert.equal(calls, 0);
  assert.equal(result.iterations, 0);
  await assert.rejects(
    conjugateGradient(async () => Float32Array.of(NaN), Float32Array.of(1), 0),
    /NaN or Infinity/,
  );
});

function quadratic(): Objective {
  return {
    evaluate: async (x) => ({
      energy: x[0] ** 2 + 2 * x[1] ** 2,
      elastic: 0,
      gradient: Float32Array.of(2 * x[0], 4 * x[1]),
    }),
    energy: async (x) => x[0] ** 2 + 2 * x[1] ** 2,
    hvp: async (_x, v) => Float32Array.of(2 * v[0], 4 * v[1]),
  };
}
test('Newton converges with residual reduction and energy descent', async () => {
  const result = await newtonCG(quadratic(), Float32Array.of(0.3, -0.2), 10);
  assert.ok(result.converged);
  assert.ok(result.residual < 0.003);
  assert.ok(result.residual < result.initialResidual);
  assert.ok(result.energy < result.initialEnergy);
});

test('line search backtracks an indefinite Newton direction, bounded work is explicit', async () => {
  const objective: Objective = {
    evaluate: async (x) => ({
      energy: (x[0] ** 2 - 1) ** 2,
      elastic: 0,
      gradient: Float32Array.of(4 * x[0] * (x[0] ** 2 - 1)),
    }),
    energy: async (x) => (x[0] ** 2 - 1) ** 2,
    hvp: async (x, v) => Float32Array.of((12 * x[0] ** 2 - 4) * v[0]),
  };
  const result = await newtonCG(objective, Float32Array.of(0.1), 10);
  assert.ok(result.negativeCurvature > 0);
  assert.ok(result.energy < result.initialEnergy);
  assert.ok(result.cgIterations <= 32);
  const limited = await newtonCG(quadratic(), Float32Array.of(1, 1), 10, {
    newton: 1,
    cg: 1,
    lineSearch: 3,
  });
  assert.equal(limited.converged, false);
  assert.equal(limited.iterations, 1);
  assert.ok(limited.energy < limited.initialEnergy);
});

test('failed line search does not mutate the source state or hide NaN', async () => {
  const x = Float32Array.of(0.2, 0.1),
    original = x.slice();
  const failing = (energy: number): Objective => {
    let calls = 0;
    const base = quadratic();
    return {
      ...base,
      evaluate: async (q) => {
        const result = await base.evaluate(q);
        return { ...result, energy: calls++ === 0 ? result.energy : energy };
      },
    };
  };
  await assert.rejects(newtonCG(failing(1e8), x, 10), /could not decrease energy/);
  assert.deepEqual(x, original);
  await assert.rejects(newtonCG(failing(NaN), x, 10), /non-finite/);
});

test('unresolvable Float32 decrease is labeled precision-limited, never converged', async () => {
  const result = await newtonCG(
    {
      evaluate: async (x) => ({
        energy: x[0] === 0 ? 1 : 1.00001,
        elastic: 0,
        gradient: Float32Array.of(0.01),
      }),
      energy: async () => 1.00001,
      hvp: async (_x, v) => Float32Array.from(v, (a) => a * 10000),
    },
    Float32Array.of(0),
    8000,
  );
  assert.equal(result.termination, 'precision');
  assert.equal(result.converged, false);
  assert.equal(result.energy, result.initialEnergy);
  assert.equal(result.x[0], 0);
});

test('a rejected speculative proposal discards the next proposal and backtracks safely', async () => {
  const base = quadratic();
  const initial = Float32Array.of(0.2, 0.1);
  const result = await newtonCG(
    {
      ...base,
      batch: async (x) => {
        const bad = Float32Array.of(100, 100);
        return {
          initial: await base.evaluate(x),
          proposals: [
            {
              candidate: bad,
              direction: Float32Array.of(99.8, 99.9),
              evaluation: await base.evaluate(bad),
              iterations: 1,
              negativeCurvature: false,
            },
            {
              candidate: Float32Array.of(NaN, NaN),
              direction: Float32Array.of(NaN, NaN),
              evaluation: { energy: NaN, elastic: NaN, gradient: Float32Array.of(NaN, NaN) },
              iterations: 1,
              negativeCurvature: false,
            },
          ],
        };
      },
    },
    initial,
    10,
  );
  assert.ok(result.backtracks > 0);
  assert.ok(result.energy < result.initialEnergy);
  assert.ok(result.x.every(Number.isFinite));
});

// Model-free tests here use a quadratic oracle, never a JS physics replacement.
const mesh: Mesh = {
  opset: 22,
  grid: 2,
  mass: 0.04,
  epsilon: 0.015,
  positions: [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ],
  edges: [],
  models: {} as Mesh['models'],
  runtime: {} as Mesh['runtime'],
};
const oracle: Pick<Graphs, 'objective'> = {
  objective: (p: Parameters) => ({
    evaluate: async (x: Vector) => ({
      energy: x.reduce(
        (sum, v, i) => sum + 0.5 * p.diagonal[i] * (v - p.target[i]) ** 2 - p.load[i] * v,
        0,
      ),
      elastic: 0,
      gradient: Float32Array.from(
        x,
        (v, i) => p.free[i] * (p.diagonal[i] * (v - p.target[i]) - p.load[i]),
      ),
    }),
    energy: async (x: Vector) =>
      oracle
        .objective(p)
        .evaluate(x)
        .then((v) => v.energy),
    hvp: async (_x: Vector, v: Vector) =>
      Float32Array.from(v, (a, i) => p.free[i] * p.diagonal[i] * a),
  }),
};

test('implicit integration preserves pins and accounts for mass, gravity and damping', async () => {
  const sim = new Simulation(mesh, oracle);
  sim.settings.dt = 0.01;
  sim.settings.damping = 2;
  sim.v[5] = 0.3;
  const expected = (0.3 + 0.01 * 9.81) / 1.02;
  await sim.step();
  assert.ok(Math.abs(sim.v[5] - expected) < 0.002);
  assert.deepEqual(Array.from(sim.x.slice(0, 4)), [0, 0, 1, 0]);
  assert.equal(sim.v[0], 0);
  sim.reset();
  assert.deepEqual(sim.x, sim.rest);
  assert.equal(sim.steps, 0);
});

test('safety limit preserves the last valid state and reports a remedy', async () => {
  const sim = new Simulation(mesh, oracle);
  sim.v[5] = 1000;
  const initial = sim.x.slice();
  await assert.rejects(sim.step(), /Simulation safety limit.*reduce timestep/i);
  assert.deepEqual(sim.x, initial);
  assert.equal(sim.steps, 0);
});

test('force arrows preserve graph-force direction and use a bounded monotone magnitude scale', () => {
  assert.equal(forceArrow(0, 0), null);
  const gravity = forceArrow(0, mesh.mass * 9.81)!;
  assert.equal(gravity.dx, 0);
  assert.equal(gravity.dy, 1);
  const diagonal = forceArrow(-3, 4)!;
  assert.equal(diagonal.dx, -0.6);
  assert.equal(diagonal.dy, 0.8);
  assert.ok(diagonal.length > gravity.length);
  assert.ok(forceArrow(1000, 0)!.length <= 36);
  assert.ok(diagonal.strength > gravity.strength);
});
