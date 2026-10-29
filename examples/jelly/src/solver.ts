// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
export type Vector = Float32Array<ArrayBuffer>;
export interface Evaluation {
  energy: number;
  elastic: number;
  gradient: Vector;
  physicalGradient?: Vector;
}
export interface Objective {
  evaluate(x: Vector): Promise<Evaluation>;
  energy(x: Vector): Promise<number>;
  hvp(x: Vector, direction: Vector): Promise<Vector>;
  cg?(x: Vector, gradient: Vector, regularization: number): Promise<CGResult>;
  propose?(x: Vector, gradient: Vector, regularization: number): Promise<Proposal>;
  batch?(
    x: Vector,
    regularization: number,
  ): Promise<{ initial: Evaluation; proposals: Proposal[] }>;
}
export interface CGResult {
  direction: Vector;
  iterations: number;
  negativeCurvature: boolean;
}
export interface Proposal extends CGResult {
  candidate: Vector;
  evaluation: Evaluation;
  energyChange?: number;
}
export interface SolveResult {
  x: Vector;
  residual: number;
  initialResidual: number;
  initialEnergy: number;
  energy: number;
  iterations: number;
  cgIterations: number;
  negativeCurvature: number;
  backtracks: number;
  converged: boolean;
  termination: 'converged' | 'budget' | 'precision';
  evaluation: Evaluation;
}

export function dot(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
export function norm(a: Vector): number {
  return Math.sqrt(dot(a, a));
}
export function finite(a: Vector, label: string): void {
  if (!a.every(Number.isFinite)) {
    throw new Error(
      `${label} contains NaN or Infinity. Pause and reset, then reduce timestep or stiffness.`,
    );
  }
}
function add(a: Vector, b: Vector, scale: number): Vector {
  return Float32Array.from(a, (value, i) => value + scale * b[i]);
}

/** Truncated CG: never divide by zero/negative curvature or hide a bad HVP. */
export async function conjugateGradient(
  multiply: (v: Vector) => Promise<Vector>,
  gradient: Vector,
  regularization: number,
  maxIterations = 8,
): Promise<CGResult> {
  let r = Float32Array.from(gradient, (value) => -value);
  let d = r.slice();
  let p = new Float32Array(r.length);
  let rr = dot(r, r);
  const threshold = Math.max(1e-12, rr * 0.02 ** 2);
  let iterations = 0;
  for (; iterations < maxIterations && rr > threshold; iterations++) {
    const raw = await multiply(d);
    finite(raw, 'Generated Hessian-vector product');
    const hd = add(raw, d, regularization);
    const curvature = dot(d, hd);
    if (curvature <= 1e-8 * dot(d, d)) {
      return {
        direction: iterations === 0 ? Float32Array.from(gradient, (value) => -value) : p,
        iterations: iterations + 1,
        negativeCurvature: true,
      };
    }
    const alpha = rr / curvature;
    p = add(p, d, alpha);
    r = add(r, hd, -alpha);
    const next = dot(r, r);
    d = add(r, d, next / rr);
    rr = next;
  }
  finite(p, 'CG direction');
  return { direction: p, iterations, negativeCurvature: false };
}

/** Matrix-free damped Newton with a step radius and Armijo safeguarding. */
export async function newtonCG(
  objective: Objective,
  initial: Vector,
  stiffness: number,
  limits = { newton: 4, cg: 8, lineSearch: 7 },
): Promise<SolveResult> {
  let x = initial.slice();
  let regularization = Math.max(0.001, stiffness * 1e-5);
  const batch =
    objective.batch && limits.cg === 8 && limits.newton >= 2
      ? await objective.batch(x, regularization)
      : undefined;
  const proposals = batch?.proposals ?? [];
  let current = batch?.initial ?? (await objective.evaluate(x));
  finite(current.gradient, 'Generated energy gradient');
  if (!Number.isFinite(current.energy))
    throw new Error(
      'Initial ONNX energy is not finite. Reset the sheet and reduce timestep or stiffness.',
    );
  const initialResidual = norm(current.gradient);
  const initialEnergy = current.energy;
  const tolerance = 0.003 + initialResidual * 0.001;
  let termination: SolveResult['termination'] = 'budget';
  let iterations = 0,
    cgIterations = 0,
    negativeCurvature = 0,
    backtracks = 0;
  for (; iterations < limits.newton && norm(current.gradient) > tolerance; iterations++) {
    const proposal =
      proposals.shift() ??
      (objective.propose && limits.cg === 8
        ? await objective.propose(x, current.gradient, regularization)
        : undefined);
    const cg =
      proposal ??
      (objective.cg && limits.cg === 8
        ? await objective.cg(x, current.gradient, regularization)
        : await conjugateGradient(
            (v) => objective.hvp(x, v),
            current.gradient,
            regularization,
            limits.cg,
          ));
    cgIterations += cg.iterations;
    if (cg.negativeCurvature) {
      negativeCurvature++;
      regularization *= 10;
    }
    let direction = cg.direction;
    if (dot(current.gradient, direction) >= 0) {
      direction = Float32Array.from(current.gradient, (value) => -value);
    }
    const radius = 0.45;
    const length = norm(direction);
    if (length > radius) direction = Float32Array.from(direction, (v) => (v * radius) / length);
    const slope = dot(current.gradient, direction);
    let accepted: Vector | undefined;
    let acceptedEvaluation: Evaluation | undefined;
    for (let line = 0; line < limits.lineSearch; line++) {
      const step = 0.5 ** line;
      const candidate = line === 0 && proposal ? proposal.candidate : add(x, direction, step);
      const evaluation =
        line === 0 && proposal ? proposal.evaluation : await objective.evaluate(candidate);
      const energy = evaluation.energy;
      if (!Number.isFinite(energy)) {
        throw new Error(
          'ONNX energy became non-finite during line search. Reset and reduce timestep or stiffness.',
        );
      }
      const sufficientDecrease =
        line === 0 && proposal?.energyChange !== undefined
          ? proposal.energyChange <= 1e-4 * slope
          : energy <= current.energy + 1e-4 * step * slope;
      if (sufficientDecrease) {
        accepted = candidate;
        acceptedEvaluation = evaluation;
        break;
      }
      backtracks++;
      proposals.length = 0;
    }
    if (!accepted) {
      // FLOAT32 energy cannot resolve arbitrarily small decreases. Do not
      // relabel stagnation as convergence or accept a known uphill trial.
      const energyResolution = 8 * 2 ** -23 * Math.max(1, Math.abs(current.energy));
      if (-slope <= energyResolution && length < 1e-4) {
        termination = 'precision';
        break;
      }
      throw new Error(
        `Newton line search could not decrease energy after ${limits.lineSearch} trials ` +
          `(residual ${norm(current.gradient).toPrecision(3)}). Last valid state preserved; ` +
          'reduce timestep or release the pointer, then resume.',
      );
    }
    x = accepted;
    current = acceptedEvaluation!;
    finite(current.gradient, 'Newton residual');
  }
  return {
    x,
    residual: norm(current.gradient),
    initialResidual,
    initialEnergy,
    energy: current.energy,
    iterations,
    cgIterations,
    negativeCurvature,
    backtracks,
    converged: norm(current.gradient) <= tolerance,
    termination: norm(current.gradient) <= tolerance ? 'converged' : termination,
    evaluation: current,
  };
}
