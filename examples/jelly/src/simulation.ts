// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import type { Graphs, Mesh, Parameters } from './model';
import { finite, newtonCG, norm, type SolveResult, type Vector } from './solver';

export interface Settings {
  stiffness: number;
  damping: number;
  dt: number;
  gravity: boolean;
  pins: 'top' | 'corners' | 'none';
  overlay: 'strain' | 'force' | 'none';
}
export class Simulation {
  x: Vector;
  v: Vector;
  acceleration: Vector;
  forces: Vector;
  readonly rest: Vector;
  readonly pinned = new Set<number>();
  readonly anchors: Vector;
  dragging: { node: number; position: [number, number] } | null = null;
  settings: Settings = {
    stiffness: 120,
    damping: 2,
    dt: 0.025,
    gravity: true,
    pins: 'top',
    overlay: 'force',
  };
  steps = 0;
  time = 0;
  stepMs = 0;
  frameSamples: number[] = [];
  elastic = 0;
  kinetic = 0;
  residual = 0;
  lastSolve: SolveResult | null = null;
  status = 'Ready to deform';

  constructor(
    readonly mesh: Mesh,
    readonly graphs: Pick<Graphs, 'objective'>,
  ) {
    this.rest = Float32Array.from(mesh.positions.flat());
    this.x = this.rest.slice();
    this.anchors = this.rest.slice();
    this.v = new Float32Array(this.x.length);
    this.acceleration = new Float32Array(this.x.length);
    this.forces = new Float32Array(this.x.length);
    this.setPins('top');
  }

  reset(): void {
    this.x = this.rest.slice();
    this.v.fill(0);
    this.acceleration.fill(0);
    this.forces.fill(0);
    this.anchors.set(this.rest);
    this.dragging = null;
    this.steps = 0;
    this.time = 0;
    this.elastic = 0;
    this.kinetic = 0;
    this.residual = 0;
    this.lastSolve = null;
    this.frameSamples.length = 0;
    this.stepMs = 0;
    this.status = 'Reset · rest configuration';
  }
  setPins(mode: Settings['pins']): void {
    this.settings.pins = mode;
    const n = this.mesh.grid;
    const next =
      mode === 'none' ? [] : mode === 'top' ? [0, n - 1] : [0, n - 1, n * (n - 1), n * n - 1];
    for (const node of next) {
      if (!this.pinned.has(node)) {
        this.anchors[2 * node] = this.x[2 * node];
        this.anchors[2 * node + 1] = this.x[2 * node + 1];
      }
      this.v[2 * node] = this.v[2 * node + 1] = 0;
      this.acceleration[2 * node] = this.acceleration[2 * node + 1] = 0;
    }
    this.pinned.clear();
    for (const node of next) this.pinned.add(node);
    if (this.dragging && this.pinned.has(this.dragging.node)) this.dragging = null;
  }

  parameters(implicit: boolean): Parameters {
    const { dt: h, damping, stiffness, gravity } = this.settings;
    const mass = this.mesh.mass;
    const inertia = implicit ? mass / (h * h) : 0;
    const viscosity = implicit ? (mass * damping) / h : 0;
    const diagonal = new Float32Array(this.x.length);
    const target = new Float32Array(this.x.length);
    const load = new Float32Array(this.x.length);
    const free = new Float32Array(this.x.length).fill(1);
    const anchor = new Float32Array(this.x.length);
    const physical_diagonal = new Float32Array(this.x.length);
    const physical_target = new Float32Array(this.x.length);
    for (let i = 0; i < this.x.length; i++) {
      const node = Math.floor(i / 2);
      const pointer = this.dragging?.node === node ? 180 : 0;
      diagonal[i] = inertia + viscosity + pointer;
      const inertialTarget = this.x[i] + h * this.v[i];
      const cursor = this.dragging?.position[i % 2] ?? 0;
      physical_diagonal[i] = pointer;
      physical_target[i] = cursor;
      target[i] =
        diagonal[i] > 0
          ? (inertia * inertialTarget + viscosity * this.x[i] + pointer * cursor) / diagonal[i]
          : 0;
      load[i] = i % 2 && gravity ? mass * 9.81 : 0;
      if (this.pinned.has(node)) {
        free[i] = 0;
        anchor[i] = this.anchors[i];
      }
    }
    return { free, anchor, diagonal, target, load, stiffness, physical_diagonal, physical_target };
  }

  async step(): Promise<void> {
    const started = performance.now();
    // Snapshot parameters for one atomic step; UI changes take effect next step.
    const settings = { ...this.settings };
    const { dt: h } = settings;
    const physical = this.parameters(false);
    const objective = this.graphs.objective(this.parameters(true));
    const predictor = Float32Array.from(this.x, (value, i) =>
      physical.free[i] ? value + h * this.v[i] + h * h * this.acceleration[i] : physical.anchor[i],
    );
    const solve = await newtonCG(objective, predictor, settings.stiffness);
    const next = solve.x;
    const velocity = Float32Array.from(next, (value, i) => (value - this.x[i]) / h);
    for (let i = 0; i < next.length; i++) {
      if (physical.free[i] === 0) {
        next[i] = physical.anchor[i];
        velocity[i] = 0;
      }
    }
    finite(next, 'Implicit step');
    finite(velocity, 'Implicit velocity');
    if (
      next.some((value) => Math.abs(value) > 5) ||
      velocity.some((value) => Math.abs(value) > 100)
    ) {
      throw new Error(
        'Simulation safety limit: ' +
          `step would exceed 5 position units or 100 units/s (k=${settings.stiffness}, dt=${h}). ` +
          'Last valid state preserved. Reduce timestep/stiffness or drag distance, then Reset and Resume.',
      );
    }
    const physicalResult = solve?.evaluation.physicalGradient
      ? { ...solve.evaluation, gradient: solve.evaluation.physicalGradient }
      : await this.graphs.objective(physical).evaluate(next);
    this.acceleration = Float32Array.from(velocity, (value, i) => (value - this.v[i]) / h);
    this.x = next;
    this.v = velocity;
    this.steps++;
    this.time += h;
    this.forces = Float32Array.from(physicalResult.gradient, (value) => -value);
    this.elastic = physicalResult.elastic;
    this.kinetic = 0.5 * this.mesh.mass * norm(velocity) ** 2;
    this.lastSolve = solve;
    this.residual = solve.residual;
    this.status = solve.converged
      ? 'Converged · safeguarded Newton-CG'
      : solve.termination === 'precision'
        ? 'Float32 precision limit · approximate step'
        : 'Iteration budget · approximate step';
    this.stepMs = performance.now() - started;
    this.frameSamples.push(this.stepMs);
    if (this.frameSamples.length > 200) this.frameSamples.shift();
  }
}
