// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import './style.css';
import { Graphs, type Mesh } from './model';
import { Renderer } from './render';
import { type Settings, Simulation } from './simulation';

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing UI element #${id}. Restore index.html and reload the demo.`);
  return found as T;
}
const scene = element<HTMLCanvasElement>('scene');
const pauseButton = element<HTMLButtonElement>('pause');
const errorBox = element('error');
const pending: (() => void)[] = [];
let simulation: Simulation, renderer: Renderer, graphs: Graphs;
let running = true,
  busy = false,
  failed = false,
  ready = false;
let pointerId: number | null = null,
  tugSteps = 0;
const history: number[] = [];
const renderStats = { frames: 0, fps: 0, updatesPerSecond: 0 };
let rateStart = performance.now(),
  rateFrames = 0,
  rateSteps = 0;
let lastDraw = 0;
let lastRefresh = 0;
const physicsQueue = new MessageChannel();
physicsQueue.port1.onmessage = () => {
  void physics();
};

function showError(error: unknown): void {
  failed = true;
  running = false;
  errorBox.hidden = false;
  errorBox.textContent = String(error instanceof Error ? error.message : error);
  element('scene-state').textContent = 'Paused · action needed below';
  pauseButton.textContent = 'Resume';
  if (!ready) {
    element('loading').querySelector('strong')!.textContent = 'The GPU could not get started';
    element('loading').querySelector('span')!.textContent =
      'See the actionable error below the playground. No physics fallback is running.';
    const loader = element('loading').querySelector<HTMLElement>('.loader');
    if (loader) loader.style.display = 'none';
    element('backend-badge').textContent = 'GPU UNAVAILABLE';
  }
}
function clearError(): void {
  failed = false;
  errorBox.hidden = true;
}
function reset(): void {
  simulation.reset();
  tugSteps = 0;
  history.length = 0;
  clearError();
  refresh();
  renderer.draw();
}
function queue(action: () => void): void {
  if (!ready) return;
  if (busy) pending.push(action);
  else {
    action();
    refresh();
    renderer.draw();
  }
}
function refresh(): void {
  if (!simulation) return;
  element('energy').innerHTML = `${simulation.elastic.toFixed(3)} <small>J*</small>`;
  let maxForce = 0;
  for (let node = 0; node < simulation.mesh.positions.length; node++) {
    if (!simulation.pinned.has(node))
      maxForce = Math.max(
        maxForce,
        Math.hypot(simulation.forces[node * 2], simulation.forces[node * 2 + 1]),
      );
  }
  element('force-max').textContent = simulation.steps
    ? `${maxForce.toFixed(3)} units`
    : 'waiting for first step';
  element('force-legend').hidden = simulation.settings.overlay !== 'force';
  element('residual').textContent = simulation.steps ? simulation.residual.toExponential(1) : '—';
  element('step-time').innerHTML = simulation.steps
    ? `${simulation.stepMs.toFixed(1)} <small>ms</small>`
    : '—';
  const solve = simulation.lastSolve;
  element('iterations').textContent = solve
    ? `${solve.iterations} Newton · ${solve.cgIterations} CG · ${solve.negativeCurvature} curvature stops`
    : 'Waiting for the first step';
  element('graph-time').textContent = graphs.baseline
    ? `grad ${graphs.timings.gradient.last.toFixed(1)} ms · H·v ${graphs.timings.hvp.last.toFixed(1)} ms`
    : `GPU Newton batch ${graphs.timings.batch.last.toFixed(1)} ms · 1 packed readback`;
  element('scene-state').textContent = failed
    ? 'Paused · action needed below'
    : !running
      ? `Paused · ${simulation.status}`
      : simulation.status;
  element('sim-clock').textContent = `t = ${simulation.time.toFixed(3)} s`;
  element('step-rate').textContent = simulation.steps
    ? `${renderStats.updatesPerSecond.toFixed(1)} updates/s · ${renderStats.fps.toFixed(0)} render fps`
    : 'Independent rendering and bounded physics updates';
  const spark = element<HTMLCanvasElement>('sparkline');
  const c = spark.getContext('2d');
  if (c) {
    const rect = spark.getBoundingClientRect();
    spark.width = Math.round(rect.width * 2);
    spark.height = 80;
    c.scale(2, 2);
    c.strokeStyle = '#a082c4';
    c.lineWidth = 1.5;
    const max = Math.max(0.01, ...history);
    c.beginPath();
    history.forEach((value, i) => {
      const x = (i / 99) * rect.width,
        y = 35 - (value / max) * 30;
      if (i) c.lineTo(x, y);
      else c.moveTo(x, y);
    });
    c.stroke();
  }
}
async function step(): Promise<void> {
  if (!ready || busy) return;
  busy = true;
  try {
    await simulation.step();
    if (tugSteps > 0 && --tugSteps === 0) simulation.dragging = null;
    history.push(simulation.elastic + simulation.kinetic);
    if (history.length > 100) history.shift();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    while (pending.length) pending.shift()!();
  }
}
function frame(timestamp: number): void {
  if (timestamp - lastDraw < 1000 / 60 - 0.5) {
    requestAnimationFrame(frame);
    return;
  }
  lastDraw = timestamp;
  if (timestamp - lastRefresh >= 200) {
    refresh();
    lastRefresh = timestamp;
  }
  renderStats.frames++;
  rateFrames++;
  if (timestamp - rateStart >= 1000) {
    const seconds = (timestamp - rateStart) / 1000;
    renderStats.fps = rateFrames / seconds;
    renderStats.updatesPerSecond = Math.max(0, simulation.steps - rateSteps) / seconds;
    rateSteps = simulation.steps;
    rateFrames = 0;
    rateStart = timestamp;
  }
  renderer.draw();
  requestAnimationFrame(frame);
}
async function physics(): Promise<void> {
  const start = performance.now();
  if (ready && running && !document.hidden) await step();
  const delay = running
    ? Math.max(0, simulation.settings.dt * 1000 - (performance.now() - start))
    : 30;
  if (delay === 0 && !document.hidden) {
    // Unlike nested zero-delay timers, MessageChannel has no 4 ms clamp.
    physicsQueue.port2.postMessage(null);
    return;
  }
  setTimeout(
    () => {
      void physics();
    },
    document.hidden ? 250 : delay,
  );
}

for (const id of ['stiffness', 'damping', 'timestep'] as const) {
  element<HTMLInputElement>(id).addEventListener('input', (event) =>
    queue(() => {
      const value = Number((event.target as HTMLInputElement).value);
      if (id === 'stiffness') {
        simulation.settings.stiffness = Math.round(10 * 800 ** (value / 100));
        element('stiffness-value').textContent = String(simulation.settings.stiffness);
      } else if (id === 'damping') {
        simulation.settings.damping = value;
        element('damping-value').textContent = `${value.toFixed(1)} s⁻¹`;
      } else {
        simulation.settings.dt = value / 1000;
        element('timestep-value').textContent = `${value} ms`;
      }
    }),
  );
}
for (const id of ['pins', 'overlay'] as const) {
  element<HTMLSelectElement>(id).addEventListener('change', (event) => {
    const value = (event.target as HTMLSelectElement).value;
    queue(() => {
      if (id === 'pins') simulation.setPins(value as Settings['pins']);
      else simulation.settings.overlay = value as Settings['overlay'];
    });
  });
}
element('gravity').addEventListener('change', (event) => {
  const checked = (event.target as HTMLInputElement).checked;
  queue(() => {
    simulation.settings.gravity = checked;
  });
});
pauseButton.addEventListener('click', () => {
  running = !running;
  if (running) clearError();
  pauseButton.textContent = running ? 'Pause' : 'Resume';
  refresh();
});
element('reset').addEventListener('click', () => queue(reset));
element('tug').addEventListener('click', () =>
  queue(() => {
    const node =
      simulation.mesh.grid * Math.floor(simulation.mesh.grid / 2) +
      Math.floor(simulation.mesh.grid / 2);
    simulation.dragging = {
      node,
      position: [simulation.x[2 * node] + 0.32, simulation.x[2 * node + 1] - 0.28],
    };
    tugSteps = 6;
    running = true;
    clearError();
    pauseButton.textContent = 'Pause';
  }),
);
element('stiff-preset').addEventListener('click', () =>
  queue(() => {
    simulation.settings.stiffness = 8000;
    simulation.settings.dt = 0.04;
    simulation.settings.damping = 2;
    element<HTMLInputElement>('stiffness').value = '100';
    element('stiffness-value').textContent = '8000';
    element<HTMLInputElement>('timestep').value = '40';
    element('timestep-value').textContent = '40 ms';
    element<HTMLInputElement>('damping').value = '2';
    element('damping-value').textContent = '2.0 s⁻¹';
    reset();
    running = true;
    pauseButton.textContent = 'Pause';
  }),
);
scene.addEventListener('pointerdown', (event) => {
  if (!ready || pointerId !== null) return;
  const position = renderer.world(event.clientX, event.clientY);
  const node = renderer.nearest(position);
  if (node === null) return;
  pointerId = event.pointerId;
  tugSteps = 0;
  scene.setPointerCapture(event.pointerId);
  simulation.dragging = { node, position };
  renderer.draw();
});
scene.addEventListener('pointermove', (event) => {
  if (event.pointerId === pointerId && simulation.dragging) {
    simulation.dragging.position = renderer.world(event.clientX, event.clientY);
    renderer.draw();
  }
});
function releasePointer(event: PointerEvent): void {
  if (event.pointerId === pointerId) {
    pointerId = null;
    simulation.dragging = null;
    renderer.draw();
  }
}
scene.addEventListener('pointerup', releasePointer);
scene.addEventListener('pointercancel', releasePointer);
scene.addEventListener('lostpointercapture', releasePointer);
scene.addEventListener('keydown', (event) => {
  if (event.code === 'Space') {
    event.preventDefault();
    pauseButton.click();
  }
});

export interface Playground {
  simulation: Simulation;
  graphs: Graphs;
  renderer: Renderer;
  step: () => Promise<void>;
  pause: () => void;
  readonly busy: boolean;
  renderStats: typeof renderStats;
}
declare global {
  interface Window {
    jelly?: Playground;
  }
}

async function initialize(): Promise<void> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}models/mesh.json?v=${__JELLY_MODEL_VERSION__}`,
  );
  if (!response.ok)
    throw new Error(
      `Mesh manifest returned HTTP ${response.status}. Serve examples/jelly with npm run dev; include public/models when building.`,
    );
  const mesh = (await response.json()) as Mesh;
  for (const kind of ['energy', 'gradient', 'hvp'] as const) {
    const section = document.createElement('section');
    const heading = document.createElement('h3');
    heading.textContent = `${kind} · ONNX opset ${mesh.opset}`;
    section.append(heading);
    for (const [label, info, folder] of [
      ['Original', mesh.models[kind], ''],
      ['Optimized runtime', mesh.runtime[kind], 'runtime/'],
    ] as const) {
      const description = document.createElement('p');
      description.textContent = `${label}: ${info.nodes} nodes · ONNX opset ${mesh.opset}`;
      const link = document.createElement('a');
      const modelUrl = new URL(
        `models/${folder}${kind}.onnx`,
        'https://www.justinchuby.com/onnax/jelly/',
      );
      const netronUrl = new URL('https://netron.app/');
      netronUrl.searchParams.set('url', modelUrl.href);
      link.href = netronUrl.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `Open ${label.toLowerCase()} ${kind}.onnx in Netron`;
      const preview = document.createElement('a');
      preview.className = 'model-graph-preview';
      preview.href = link.href;
      preview.target = link.target;
      preview.rel = link.rel;
      preview.setAttribute('aria-label', link.textContent);
      const image = document.createElement('img');
      image.src = `${import.meta.env.BASE_URL}models/${folder}${kind}.svg?v=${__JELLY_MODEL_VERSION__}`;
      image.alt = `${label} ${kind} computation graph exported from Netron`;
      image.loading = 'lazy';
      preview.append(image);
      section.append(description, link, preview);
    }
    element('model-graphs').append(section);
  }
  graphs = new Graphs(mesh);
  simulation = new Simulation(mesh, graphs);
  renderer = new Renderer(scene, simulation);
  await graphs.initialize();
  const objective = graphs.objective(simulation.parameters(false));
  await objective.energy(simulation.x);
  const warmup = await objective.evaluate(simulation.x);
  await objective.hvp(
    simulation.x,
    Float32Array.from(simulation.x, () => 0.1),
  );
  await objective.cg?.(simulation.x, warmup.gradient, 0.001);
  await objective.propose?.(simulation.x, warmup.gradient, 0.001);
  await objective.batch?.(simulation.x, 0.001);
  ready = true;
  element('loading').hidden = true;
  element('backend-badge').textContent = graphs.fallbackAdapter
    ? 'WEBGPU · SOFTWARE ADAPTER'
    : 'WEBGPU · GPU-ONLY GRAPHS';
  element('device-info').textContent =
    `${graphs.deviceLabel} · ${graphs.fallbackAdapter ? 'software adapter' : 'hardware adapter'} · ` +
    `FLOAT32 · ONNX opset ${mesh.opset} · ORT Web 1.31.1.`;
  for (const id of ['pause', 'reset', 'tug', 'stiff-preset'])
    element<HTMLButtonElement>(id).disabled = false;
  // Keep the logarithmic slider in agreement with the initial material setting.
  element<HTMLInputElement>('stiffness').value = String(
    (Math.log(simulation.settings.stiffness / 10) / Math.log(800)) * 100,
  );
  graphs.device.lost.then((info) =>
    showError(
      new Error(
        `WebGPU device lost (${info.reason}): ${info.message}. Reload the page to recreate the ONNX sessions; update your graphics driver if this repeats.`,
      ),
    ),
  );
  graphs.device.addEventListener('uncapturederror', (event) =>
    showError(
      new Error(
        `WebGPU error: ${event.error.message}. Pause and reload; inspect the browser console for the failing kernel.`,
      ),
    ),
  );
  window.jelly = {
    simulation,
    graphs,
    renderer,
    renderStats,
    step,
    pause: () => {
      running = false;
      pauseButton.textContent = 'Resume';
      refresh();
    },
    get busy() {
      return busy;
    },
  };
  refresh();
  renderer.draw();
  requestAnimationFrame(frame);
  void physics();
}
void initialize().catch(showError);
