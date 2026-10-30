# Soft spot: an onnax WebGPU experiment

A draggable 2D spring sheet demonstrating why second derivatives matter.
Generated gradients and Hessian-vector products drive implicit backward Euler
in a safeguarded, matrix-free Newton-CG solve.

**[Open the hosted playground](https://www.justinchuby.com/onnax/jelly/)**
or visit the [documentation site](https://www.justinchuby.com/onnax/).

## Run

Requires Node.js 24+ and a current Chrome or Edge with WebGPU and graphics
acceleration. Serve over localhost or HTTPS. **No Python is needed to run**:
the small exported ONNX models and mesh manifest are checked in.

```shell
cd examples/jelly
npm ci
npm run dev
```

Open the localhost URL printed by Vite (normally `http://127.0.0.1:5173`).
Drag any unpinned node with a pointer or touch. The **Tug sheet** button provides
a keyboard-accessible tether, and Space on the focused canvas toggles pause.
Change anchor points, gravity, stiffness, damping, timestep, and strain/force
overlays. Reset restores the rest state without changing the controls.

Select **Try the stiff-spring test** for stiffness 8000 and timestep 40 ms.
The implicit solver displays residuals and any precision/iteration limit.
The former explicit selector was removed: symplectic Euler did use the onnax
gradient, but not the HVP, and the shared 25 ms default was numerically unstable
in that mode. It is not needed to demonstrate the implicit solver.

**ONNX force arrows are visible by default.** Drag a node or use **Tug sheet**;
mint-to-gold color and nonlinear arrow length show increasing force magnitude.
The legend reports the largest free-node force. **Show me** switches to strain
or hides overlays. Expand **How the computation works** for real energy,
gradient, and HVP graph excerpts, tensor interfaces, node counts, and model links.

```shell
npm run build
npm run preview
```

`dist/` is static-host-ready. The default build targets a site's root path. For a subpath, use
`npm run build -- --base=/your-path/`. Vite includes the pinned ORT JSEP
WASM runtime locally (about 29 MB uncompressed); there is no CDN dependency.
WASM hosts the runtime, **not a fallback physics execution provider**.

## What is actually differentiated?

The fixed mesh has **36 nodes, 72 coordinates, and 110 springs**: horizontal,
vertical, and both cell diagonals. Each node has mass 0.04. Positions start on
a square from -0.65 to 0.65; positive y points down. For incidence matrix B,
spring rest lengths are measured in that configuration with the same 0.015
length regularizer used at runtime:

```text
x = free * q + anchor
length[e] = sqrt(sum((B @ x)[e] ** 2) + epsilon ** 2)
elastic = stiffness / 2 * sum((length - rest_length) ** 2)
E = elastic + 1/2 * sum(diagonal * (x - target) ** 2) - sum(load * x)
```

`free` is zero on pinned coordinates; `anchor` stores their fixed positions.
This projects the gradient and HVP onto free degrees of freedom exactly.
Changing pin mode anchors newly pinned nodes at their current location; reset
anchors them at rest. Gravity contributes `load.y = mass * 9.81`.
Dragging adds a quadratic tether of stiffness 180 to the selected node.

For physical force overlays, `diagonal` contains only the pointer tether and
`force = -grad(E)`. The gradient output is negated, not recomputed in JavaScript.
Arrow direction is the free-node force direction; length is
`6 + 30 tanh(|force| / 6)` CSS pixels above 0.02 illustrative force units.
Color uses the same relative magnitude scale. Pinned reactions and viscous drag
are excluded; the readback represents the last completed step.

The implicit objective adds `mass / h**2` and `mass * damping / h` to the diagonal.
Its target is the diagonal-weighted combination of `x_old + h * v_old`,
`x_old`, and the cursor. Minimizing this incremental energy is backward Euler
with the **same mass, gravity, conservative spring/tether forces, and viscous
damping**, not a separate physical model.

[`export_models.py`](export_models.py) constructs this scalar energy with
`onnx_ir`, calls `onnax.grad(..., inputs=["q"], output="energy")`, then
`onnax.jvp(gradient, inputs=["q"])`. The browser loads:

| Model | Outputs |
|---|---|
| `energy.onnx` | Elastic and total incremental/physical energy |
| `gradient.onnx` | Above plus generated `q_cotangent` |
| `hvp.onnx` | Above plus generated `q_cotangent_tangent`; seed `q_tangent` |

These are ordinary FLOAT32 ONNX opset-22 graphs, authored, differentiated,
optimized, checked, and executed at 22. This is a temporary, explicit browser
compatibility choice: ORT Web 1.31.1 JSEP lacks the `Reshape`/`Unsqueeze` schema
registrations introduced at 23, and rejects these valid graphs at 23/24 with CPU
fallback disabled. The native WebGPU Asyncify runtime accepts those schemas but
fails GPU-buffer graph-capture replay in this setup. No runtime import relabeling
or CPU physics fallback is used. Other runnable Python examples use opset 24. Their
arithmetic uses the appropriate supported standard operators; no unrelated newer
operator is inserted merely to change the operator list. There are **no handwritten
runtime derivatives or dense Hessians**. Independent analytic derivatives exist
only in the Python tests.

For concrete broadcast shapes, onnax skips unnecessary reductions or emits
fixed reduction axes directly. Unknown/symbolic shapes retain general runtime
shape algebra, including `NonZero`. The exporter evaluates only constant subgraphs
and concrete `Shape` and `Size` queries, then applies the **ONNXScript optimizer**.
No ONNX Runtime optimizer is used in export, and browser sessions set
`graphOptimizationLevel: 'disabled'`. Runtime-dependent derivative arithmetic
stays intact during shape specialization. A fixed incidence `MatMul` avoids index/scatter
derivatives. Shape inference uses **onnx-shape-inference**, and source/generated/
specialized models pass IR `CheckerPass(full_check=True)` and ONNX full checking.
No library transformation guarantees or unsupported-case errors are changed.

### Regenerate the checked-in models

Use Python 3.11+ in an isolated environment, from the repository root:

```shell
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e .
python -m pip install -r examples/jelly/requirements-export.txt
python examples/jelly/export_models.py
python -m pip install pytest
python -m pytest examples/jelly/tests/test_models.py -q
```

Keep the canonical models, `runtime/` models, and `mesh.json` together. The manifest records
their actual generated interfaces; startup rejects mismatches. Pinning the export
toolchain makes regeneration reproducible; floating-point artifacts can differ
if the toolchain changes, so rerun the independent checks.

## Solver safeguards and honest limits

- At most **4 Newton iterations**, each with **8 CG iterations** and **7
  backtracking trials** per simulation step. Rendering is independent, capped
  at 60 fps. Physics is paced by its timestep and slows if the device cannot
  keep up; there is no unbounded catch-up loop.
- CG operates only through the exported HVP. A small positive diagonal
  regularizer, negative-curvature truncation, a 0.45 step-radius cap, and an
  Armijo line search handle indefinite compressed-spring Hessians.
- Residuals are displayed. Converged, iteration-budget, and FLOAT32
  precision-limited steps have distinct statuses. A tiny unresolvable predicted
  decrease can stop at the current iterate **without claiming convergence** or
  accepting an uphill line-search trial.
- The initial guess extrapolates position, velocity, and the previous
  acceleration; this is only a warm start, not a different integration method.
  Full-step Armijo tests use an algebraically identical, cancellation-resistant
  energy difference rather than subtracting large FLOAT32 energy totals.
- Failed searches and non-finite graph outputs pause with a remedy. Neither mode
  silently clamps coordinates/velocities, resets unstable states, or hides errors.
  A proposed position beyond 5 units or velocity beyond 100 units/s is rejected,
  retaining the last valid simulation state.

This is an illustrative coarse spring network, not calibrated continuum
elasticity. It has no collision detection, self-contact, or volume preservation.
Force arrows show conservative free-DOF spring + gravity + tether forces from
the last completed step, not constraint reactions or viscous drag; their display
length uses a nonlinear scale. Strain coloring shows geometric extension relative
to the rest shape. The energy history is elastic + kinetic energy, **not**
the incremental objective or a conservation claim with external forces.

## Execution and performance

The app imports `onnxruntime-web/all` with the matching JSEP WASM host, reuses
ORT's shared default `GPUDevice`, and
sets `session.disable_cpu_ep_fallback=1`. Missing WebGPU, incompatible graphs,
device loss, and graph failures are explicit errors: there is no automatic WASM
simulation fallback. The adapter identity and whether it is a software/fallback
adapter appear in Execution details.

The pinned ORT 1.31.1 native `/webgpu` Asyncify build was also tested. Its custom
device IO binding and subsequent graph-capture replay failed in this setup.
The JSEP **WebGPU** build supports the required kernels, IO binding, and capture;
this is not a switch to the WASM execution provider.

The exporter additionally applies ONNXScript optimization, removes redundant
singleton reductions and broadcast copies, prunes dead primal outputs, and
hoists tangent-independent coefficients out of linear HVP arithmetic. Canonical
autodiff metadata remains intact; packed runtime graphs explicitly record their
source interfaces instead of advertising an incompatible transform interface.
Reassociation changes FLOAT32 rounding and is tested against the canonical graphs.

| Graph | Canonical nodes | Optimized runtime nodes |
|---|---:|---:|
| Energy | 21 | 21 |
| Gradient | 41 | 48 (also supplies physical force for overlays) |
| HVP | 68 | 41 |

The UI's **Original** graphs have already undergone static shape specialization
and ONNXScript optimization. They retain the canonical derivative interfaces.
**Optimized runtime** graphs additionally specialize outputs, compose the physical
gradient, and factor linear HVP expressions for repeated execution.

At a fixed Newton position `q`, the HVP is linear in the CG direction `v`.
For example, `(a(q) * v) * b(q) + c(q) * v` can be factored into
`(a(q) * b(q) + c(q)) * v`. All multiplications in this example are elementwise;
the exporter does not commute arbitrary matrix products. The public
`onnax.HoistLinearCoefficientsPass(inputs=["q_tangent"])` performs this algebra on
the generated HVP, and ONNXScript CSE on the unrolled CG graph can share
the direction-independent coefficient work across its eight direction updates.
Those coefficients still depend on runtime positions and physical parameters:
they must be recomputed when those inputs change. This is neither a precomputed
dense Hessian nor a handwritten spring derivative.

The runtime gradient composes the same generated gradient with physical-only
diagonal/target inputs. CSE shares spring arithmetic, avoiding a separate
post-solve force evaluation without handwriting a replacement derivative.
`cg.onnx` unrolls eight safeguarded matrix-free CG slots, keeping reductions,
directions, residuals, and curvature decisions on GPU. `step.onnx` composes that
graph with a bounded descent proposal and its generated gradient.
`batch.onnx` speculates **two** such Newton proposals in one captured run.
The CPU checks each proposal's Armijo condition in order, discards subsequent
speculation after a rejection, backtracks normally, and can continue to the
unchanged four-Newton budget.

Initializers, reused input/output buffers, and cached capture commands remain
GPU-resident. Each step uploads its parameter snapshot and positions; the
common implicit path returns one packed buffer (about 2.4 KB). Rendering reads
positions through that buffer. This is **not zero-copy**: CPU line-search
decisions and Canvas 2D rendering still require a readback. A batch executes
16 fixed CG slots even when the displayed *effective* iteration count is lower;
masked slots cannot change a stopped CG result. Timing JSON records both.

### Measured hardware performance

The measurements below are historical results from the earlier export pipeline
with ONNX Runtime graph optimization enabled. They have not been remeasured for
the current ONNXScript-only pipeline and disabled runtime graph optimization.

On Intel **gen-12lp**, hardware adapter (not a fallback), Windows Edge 157,
the production build was tested with the unchanged 6×6 mesh, stiffness 120,
damping 2, timestep 25 ms, and a continuously moving pointer tether. Each trial
separates **2 s warmup** from **6 s sustained sampling**; three trials were run.
Rendering and physics rates are measured separately:

| Execution | Physics updates/s | Render fps | Simulated time / wall time |
|---|---:|---:|---:|
| Canonical graphs, CPU feeds/readbacks and per-HVP CG decisions | 4.8–5.0 | ~60 | 0.12× |
| Optimized, clock-paced default | 37.6–39.5 | ~60 | 0.94–0.99× |
| Same optimized solver, uncapped benchmark pump | 46.8–53.6 | ~60 | 1.17–1.34× |

The default intentionally does not advance physics faster than its clock.
The uncapped test measures actual completed physics updates, **not render fps**;
it retains the same mesh, timestep, tolerances, work budgets and moving tether.
These results are device/workload-specific, not a promise that every adapter,
large drag, or stiff material runs at that rate. The stiff 8000/40 ms test
remains stable and displays any residual/budget/precision limit.

Specialized GPU XPBD demos, such as [45deg's 3D jelly](https://45deg.github.io/jelly/),
use a different constraint solver and workload; their fps is not an
apples-to-apples Newton-CG/HVP comparison.

The UI shows whole-step and graph-call **wall time**, not GPU-only kernel time.
Benchmark JSON includes per-step solver counts/residuals, standalone energy/
gradient/HVP calls, graph submission times, and GPU-completion/readback waits.
The latter includes queued kernel execution and browser/driver scheduling:
it is not a measurement of transfer bandwidth alone.

## Validation

From `examples/jelly`:

```shell
npm run lint
npm test
npm run build
npm run test:browser
npm run benchmark -- --repeat-each=3
```

Browser tests default to installed Edge on Windows and Playwright Chromium
elsewhere. Install Chromium with `npx playwright install chromium` if needed;
set `JELLY_BROWSER=chromium` using your shell to select it explicitly.
A real WebGPU adapter is required for the GPU smoke test. Software adapters,
when used, are reported and are not hardware-performance evidence.

The browser suite records **actual WebGPU compute dispatches at capture time**
and queue submissions for energy/gradient/HVP replay, verifies finite-difference derivatives on that
backend, verifies physical force sign/pin projection and stiff implicit residual/descent,
and exercises pointer/touch, all controls, pause/reset, resize, overlays, and
unsupported-browser diagnostics. It saves desktop/mobile screenshots and a
JSON timing/device attachment under `test-results/`.

Python tests independently check analytic/finite-difference force and HVP,
HVP symmetry, exact pin projection, rest/deformed/compressed/collapsed/stiff
states, source-to-specialized numerical equivalence, and serialized interfaces.
ONNX reference evaluation checks the composed CG/proposal/batch arithmetic,
including negative curvature, zero gradients, and stable energy differences
against independent FLOAT64 energies.
Unit tests cover CG negative curvature, residual/descent, bounded budgets,
precision-limited stopping, failure rollback, mass, gravity, and damping.
CI runs the CPU/model checks and web lint/unit/build checks; GPU/browser smoke
is explicit local validation rather than pretending hosted CI has hardware GPU.

To test the static production build and enforce the measured hardware target,
build first, then set `JELLY_PREVIEW=1` and `JELLY_PERF_REQUIRED=1` in your shell
before running the browser tests/benchmark. For example, in PowerShell:

```powershell
npm run build
$env:JELLY_PREVIEW = '1'
$env:JELLY_PERF_REQUIRED = '1'
npm run test:browser
```

Enforcement requires at least 30 clock-paced updates/s, 45 render fps, and
40 uncapped updates/s (at least real-time simulated-time throughput).
Do not treat a software adapter's results as hardware performance evidence.

## Documentation deployment

The `Deploy docs` workflow builds on Node.js 24 using `npm ci`, preserves `docs/`
at the site root, adds the Vite build at `/jelly/`, and publishes an official
GitHub Pages artifact. Only trusted `main` builds can deploy to the
`github-pages` environment; pull requests build and validate but cannot deploy.
The repository inherits the owner's existing domain; no domain settings are
replaced. The site uses one WASM host thread and no proxy worker, so it does not
require cross-origin isolation headers or `SharedArrayBuffer` on Pages.
The matching JSEP JavaScript host is bundled in the main JavaScript asset.

Build the same artifact locally from the repository root:

```shell
npm ci --prefix examples/jelly
npm run build --prefix examples/jelly -- --base=/onnax/jelly/
node tools/build_docs.mjs /onnax
```

`_site/` includes the landing page, preserved static docs, `.nojekyll`, all nine
ONNX models, manifest, and pinned ORT WASM/JS. Assembly rejects missing, empty,
mixed-version models and incorrectly based HTML. Vite also fails before building
if any checked-in model is missing. A content digest of the complete model set
versions model/manifest requests and the JavaScript bundle to avoid stale model
mixing after deployment. The computation panel displays checked-in SVG diagrams
exported by Netron 9.3.3 from the original and runtime energy/gradient/HVP models.
Links open the published models in Netron. After regenerating these ONNX files,
open each in Netron and use **File → Export as SVG**, saving beside its model
with the same basename. Update that ONNX file’s SHA-256 in
`public/models/netron-sources.json` after refreshing the SVG. The build rejects
changed models until their diagram provenance is updated. The previews do not execute models.

Run the browser suite against the deployed URL without a local server:

```powershell
$env:JELLY_BASE_URL = 'https://www.justinchuby.com/onnax/jelly/'
npm run test:browser --prefix examples/jelly
```

For local production testing at the project subpath, set `JELLY_PREVIEW=1` and
`JELLY_BASE_PATH=/onnax/jelly/`; the suite starts Vite preview on port 5175.
