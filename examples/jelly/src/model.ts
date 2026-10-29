// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT

import * as ort from 'onnxruntime-web/all';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import type { Evaluation, Objective, Proposal, Vector } from './solver';
import { finite } from './solver';

type Kind = 'energy' | 'gradient' | 'hvp' | 'cg' | 'step' | 'batch';
export interface ModelInfo {
  inputs: string[];
  outputs: string[];
  operators: string[];
  nodes: number;
  graph: {
    inputs: string[];
    outputs: string[];
    excerpt: string[];
  };
}
export interface Mesh {
  opset: number;
  grid: number;
  mass: number;
  epsilon: number;
  positions: [number, number][];
  edges: [number, number][];
  models: Record<'energy' | 'gradient' | 'hvp', ModelInfo>;
  runtime: Record<Kind, ModelInfo>;
}
export interface Parameters {
  free: Vector;
  anchor: Vector;
  diagonal: Vector;
  target: Vector;
  load: Vector;
  stiffness: number;
  physical_diagonal: Vector;
  physical_target: Vector;
}
export interface Metrics {
  count: number;
  last: number;
  samples: number[];
  submitMs?: number;
  readbackMs?: number;
  submitSamples?: number[];
  readbackSamples?: number[];
  totalMs?: number;
  totalSubmitMs?: number;
  totalReadbackMs?: number;
}

export class Graphs {
  readonly timings: Record<Kind, Metrics> = {
    energy: { count: 0, last: 0, samples: [] },
    gradient: { count: 0, last: 0, samples: [] },
    hvp: { count: 0, last: 0, samples: [] },
    cg: { count: 0, last: 0, samples: [] },
    step: { count: 0, last: 0, samples: [] },
    batch: { count: 0, last: 0, samples: [] },
  };
  deviceLabel = '';
  fallbackAdapter = false;
  timestampQueries = false;
  device!: GPUDevice;
  readonly baseline = new URLSearchParams(location.search).has('baseline');
  private sessions!: Partial<Record<Kind, ort.InferenceSession>>;
  private readonly inputs = new Map<string, { buffer: GPUBuffer; tensor: ort.Tensor }>();
  private readonly outputs = new Map<
    string,
    { buffer: GPUBuffer; tensor: ort.Tensor; staging: GPUBuffer; bytes: number }
  >();
  private cachedParameters: Parameters | undefined;

  constructor(readonly mesh: Mesh) {}

  async initialize(): Promise<void> {
    if (!navigator.gpu)
      throw new Error(
        'WebGPU is unavailable. Open this page in a current Chrome or Edge with hardware acceleration, over HTTPS or localhost. No WASM physics fallback is enabled.',
      );
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter)
      throw new Error(
        'No WebGPU adapter was found. Enable hardware acceleration, update your graphics driver, and reload in Chrome or Edge.',
      );
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.initTimeout = 30_000;
    ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, location.href).href };
    const sessions: Partial<Record<Kind, ort.InferenceSession>> = {};
    try {
      const kinds: Kind[] = this.baseline
        ? ['energy', 'gradient', 'hvp']
        : ['energy', 'gradient', 'hvp', 'cg', 'step', 'batch'];
      for (const name of kinds) {
        const url = `${import.meta.env.BASE_URL}models/${this.baseline ? '' : 'runtime/'}${name}.onnx?v=${__JELLY_MODEL_VERSION__}`;
        const response = await fetch(url);
        if (!response.ok)
          throw new Error(
            `Model ${url} returned HTTP ${response.status}. Run npm ci and npm run dev from examples/jelly; keep public/models in the static build.`,
          );
        sessions[name] = await ort.InferenceSession.create(await response.arrayBuffer(), {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          enableGraphCapture: !this.baseline,
          ...(!this.baseline ? { preferredOutputLocation: 'gpu-buffer' as const } : {}),
          extra: { session: { disable_cpu_ep_fallback: '1' } },
        });
        // Use ORT's shared default device (device ordinal 0) for IO binding.
        // A custom device in ORT 1.31.1 receives ordinal 1, but external tensor
        // memory descriptors use ordinal 0 and cannot bind to that session.
        this.device = await ort.env.webgpu.device;
        const info = this.device.adapterInfo;
        this.fallbackAdapter = info.isFallbackAdapter;
        this.deviceLabel =
          [info.vendor, info.architecture, info.device, info.description]
            .filter(Boolean)
            .join(' · ') || 'WebGPU adapter (identity hidden by browser)';
        this.timestampQueries = this.device.features.has('timestamp-query');
        const expected =
          name === 'cg' || name === 'step' || name === 'batch' || !this.baseline
            ? this.mesh.runtime[name]
            : this.mesh.models[name];
        if (
          JSON.stringify(sessions[name].inputNames) !== JSON.stringify(expected.inputs) ||
          JSON.stringify(sessions[name].outputNames) !== JSON.stringify(expected.outputs)
        ) {
          throw new Error(
            `The ${name} model interface does not match mesh.json. Regenerate all artifacts together with python examples/jelly/export_models.py.`,
          );
        }
        if (!this.baseline) {
          for (const input of expected.inputs) {
            if (this.inputs.has(input)) continue;
            const dims =
              input === 'stiffness' || input === 'regularization'
                ? []
                : [this.mesh.positions.length, 2];
            const buffer = this.device.createBuffer({
              size: Math.max(16, dims.reduce((a, b) => a * b, 1) * 4),
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            this.inputs.set(input, {
              buffer,
              tensor: ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims }),
            });
          }
          const length =
            name === 'batch'
              ? this.mesh.positions.length * 16 + 12
              : name === 'step'
                ? this.mesh.positions.length * 6 + 5
                : name === 'energy'
                  ? 1
                  : this.mesh.positions.length * (name === 'gradient' ? 4 : 2) +
                    (name === 'gradient' || name === 'cg' ? 2 : 0);
          const bytes = length * 4;
          const buffer = this.device.createBuffer({
            size: Math.max(16, bytes),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          const staging = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          this.outputs.set(name, {
            buffer,
            staging,
            bytes,
            tensor: ort.Tensor.fromGpuBuffer(buffer, {
              dataType: 'float32',
              dims:
                name === 'energy'
                  ? []
                  : name === 'hvp'
                    ? [this.mesh.positions.length, 2]
                    : [length],
            }),
          });
        }
      }
    } catch (error) {
      await Promise.all(Object.values(sessions).map((session) => session.release()));
      throw new Error(
        `GPU-only ONNX model initialization failed: ${String(error)}. Check the browser console for the operator or device error; update Chrome/Edge and graphics drivers. CPU fallback is deliberately disabled.`,
        { cause: error },
      );
    }
    this.sessions = sessions;
  }

  async run(
    kind: 'energy' | 'gradient' | 'hvp',
    q: Vector,
    parameters: Parameters,
    direction?: Vector,
  ): Promise<number | Evaluation | Vector> {
    finite(q, 'Positions');
    if (!this.baseline) return this.runBound(kind, q, parameters, direction);
    const feeds: Record<string, ort.Tensor> = {};
    for (const [name, data] of Object.entries({ q, ...parameters })) {
      if (!this.mesh.models[kind].inputs.includes(name)) continue;
      feeds[name] =
        typeof data === 'number'
          ? new ort.Tensor('float32', Float32Array.of(data), [])
          : new ort.Tensor('float32', data, [this.mesh.positions.length, 2]);
    }

    if (direction)
      feeds[this.mesh.models.hvp.inputs.at(-1)!] = new ort.Tensor('float32', direction, [
        this.mesh.positions.length,
        2,
      ]);
    const names = this.mesh.models[kind].outputs;
    const requested = kind === 'hvp' ? [names.at(-1)!] : kind === 'energy' ? [names[1]] : names;
    const start = performance.now();
    let output: ort.InferenceSession.OnnxValueMapType | undefined;
    try {
      output = await this.sessions[kind]!.run(feeds, requested);
      const elapsed = performance.now() - start;
      const metric = this.timings[kind];
      metric.count++;
      metric.last = elapsed;
      metric.totalMs = (metric.totalMs ?? 0) + elapsed;
      metric.samples.push(elapsed);
      if (metric.samples.length > 200) metric.samples.shift();
      if (kind === 'hvp') {
        const result = Float32Array.from(output[names.at(-1)!].data as Float32Array);
        finite(result, 'ONNX HVP output');
        return result;
      }
      const energy = Number(output[names[1]].data[0]);
      if (!Number.isFinite(energy))
        throw new Error('The ONNX energy is not finite. Reset and reduce timestep or stiffness.');
      if (kind === 'energy') return energy;
      const gradient = Float32Array.from(output[names[2]].data as Float32Array);
      finite(gradient, 'ONNX gradient output');
      return { energy, elastic: Number(output[names[0]].data[0]), gradient };
    } finally {
      for (const tensor of Object.values(output ?? {})) tensor.dispose();
      for (const tensor of Object.values(feeds)) tensor.dispose();
    }
  }

  private async runBound(
    kind: Kind,
    q: Vector,
    parameters: Parameters,
    direction?: Vector,
    regularization?: number,
  ): Promise<number | Evaluation | Vector> {
    const start = performance.now();
    const upload = (name: string, array: Vector): void => {
      const input = this.inputs.get(name);
      if (input) this.device.queue.writeBuffer(input.buffer, 0, array);
    };
    upload('q', q);
    if (parameters !== this.cachedParameters) {
      for (const [name, data] of Object.entries(parameters)) {
        upload(name, typeof data === 'number' ? Float32Array.of(data) : data);
      }
      this.cachedParameters = parameters;
    }
    if (direction)
      upload(
        kind === 'cg' || kind === 'step' ? 'gradient' : this.mesh.models.hvp.inputs.at(-1)!,
        direction,
      );
    if (regularization !== undefined) upload('regularization', Float32Array.of(regularization));
    const feeds = Object.fromEntries(
      this.mesh.runtime[kind].inputs.map((name) => [name, this.inputs.get(name)!.tensor]),
    );
    const output = this.outputs.get(kind)!;
    await this.sessions[kind]!.run(feeds, { packed: output.tensor });
    const submitted = performance.now();
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(output.buffer, 0, output.staging, 0, output.bytes);
    this.device.queue.submit([encoder.finish()]);
    await output.staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(output.staging.getMappedRange()).slice();
    output.staging.unmap();
    finite(data, `GPU-bound ONNX ${kind} output`);
    const metric = this.timings[kind];
    metric.count++;
    metric.last = performance.now() - start;
    metric.submitMs = submitted - start;
    metric.readbackMs = performance.now() - submitted;
    metric.totalMs = (metric.totalMs ?? 0) + metric.last;
    metric.totalSubmitMs = (metric.totalSubmitMs ?? 0) + metric.submitMs;
    metric.totalReadbackMs = (metric.totalReadbackMs ?? 0) + metric.readbackMs;
    metric.submitSamples ??= [];
    metric.readbackSamples ??= [];
    metric.submitSamples.push(metric.submitMs);
    metric.readbackSamples.push(metric.readbackMs);
    if (metric.submitSamples.length > 200) metric.submitSamples.shift();
    if (metric.readbackSamples.length > 200) metric.readbackSamples.shift();
    metric.samples.push(metric.last);
    if (metric.samples.length > 200) metric.samples.shift();
    if (kind === 'hvp' || kind === 'cg' || kind === 'step' || kind === 'batch') return data;
    if (kind === 'energy') return data[0];
    const n = this.mesh.positions.length * 2;
    return {
      elastic: data[0],
      energy: data[1],
      gradient: data.slice(2, n + 2),
      physicalGradient: data.slice(n + 2),
    };
  }

  objective(parameters: Parameters): Objective {
    const n = this.mesh.positions.length * 2;
    const evaluation = (output: Vector): Evaluation => ({
      elastic: output[0],
      energy: output[1],
      gradient: output.slice(2, n + 2),
      physicalGradient: output.slice(n + 2, 2 * n + 2),
    });
    const proposal = (output: Vector, x: Vector): Proposal => {
      const candidate = output.slice(0, n);
      return {
        candidate,
        direction: Float32Array.from(candidate, (v, i) => v - x[i]),
        evaluation: evaluation(output.slice(n, 3 * n + 2)),
        energyChange: output[3 * n + 2],
        iterations: output[3 * n + 3],
        negativeCurvature: output[3 * n + 4] > 0.5,
      };
    };
    return {
      evaluate: async (x) => (await this.run('gradient', x, parameters)) as Evaluation,
      energy: async (x) => (await this.run('energy', x, parameters)) as number,
      hvp: async (x, v) => (await this.run('hvp', x, parameters, v)) as Vector,
      ...(!this.baseline
        ? {
            cg: async (x: Vector, gradient: Vector, regularization: number) => {
              const output = (await this.runBound(
                'cg',
                x,
                parameters,
                gradient,
                regularization,
              )) as Vector;
              const n = this.mesh.positions.length * 2;
              return {
                direction: output.slice(0, n),
                iterations: output[n],
                negativeCurvature: output[n + 1] > 0.5,
              };
            },
            propose: async (x: Vector, gradient: Vector, regularization: number) => {
              const output = (await this.runBound(
                'step',
                x,
                parameters,
                gradient,
                regularization,
              )) as Vector;
              return proposal(output, x);
            },
            batch: async (x: Vector, regularization: number) => {
              const output = (await this.runBound(
                'batch',
                x,
                parameters,
                undefined,
                regularization,
              )) as Vector;
              const offset = 2 * n + 2,
                width = 3 * n + 5;
              const first = proposal(output.slice(offset, offset + width), x);
              return {
                initial: evaluation(output.slice(0, offset)),
                proposals: [first, proposal(output.slice(offset + width), first.candidate)],
              };
            },
          }
        : {}),
    };
  }
}
