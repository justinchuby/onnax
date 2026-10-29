// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import { writeFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type {} from '../src/main';

test('sustained baseline vs optimized WebGPU performance', async ({
  browser,
  baseURL,
}, testInfo) => {
  const results = [];
  for (const { baseline, paced } of [
    { baseline: true, paced: true },
    { baseline: false, paced: true },
    { baseline: false, paced: false },
  ]) {
    const context = await browser.newContext({ baseURL });
    const page = await context.newPage();
    try {
      await page.goto(baseline ? './?baseline=1' : './');
      await page.waitForFunction(
        () => window.jelly || !document.querySelector<HTMLElement>('#error')!.hidden,
      );
      await expect(page.locator('#error')).toBeHidden();
      await page.evaluate(() => window.jelly!.pause());
      await page.waitForFunction(() => !window.jelly!.busy);
      const result = await page.evaluate(
        async ({ paced }) => {
          const { simulation: s, graphs, renderStats } = window.jelly!;
          s.reset();
          let sampling = false;
          const stepSamples: number[] = [];
          const solverSamples: {
            newton: number;
            cg: number;
            residual: number;
            termination: string;
          }[] = [];
          const originalStep = s.step.bind(s);
          s.step = async () => {
            await originalStep();
            if (sampling) {
              stepSamples.push(s.stepMs);
              if (s.lastSolve)
                solverSamples.push({
                  newton: s.lastSolve.iterations,
                  cg: s.lastSolve.cgIterations,
                  residual: s.lastSolve.residual,
                  termination: s.lastSolve.termination,
                });
            }
          };
          const node = 21;
          const started = performance.now();
          let driving = true;
          function drive(now: number): void {
            if (!driving) return;
            const t = (now - started) / 1000;
            s.dragging = {
              node,
              position: [
                s.rest[2 * node] + 0.12 * Math.sin(t * 1.5),
                s.rest[2 * node + 1] - 0.08 * Math.cos(t * 1.5),
              ],
            };
            requestAnimationFrame(drive);
          }
          requestAnimationFrame(drive);
          let pumping = true;
          const pump = async (): Promise<void> => {
            while (pumping && document.querySelector<HTMLElement>('#error')!.hidden) {
              await window.jelly!.step();
            }
          };
          if (paced) document.getElementById('pause')!.click();
          const pumpingDone = paced ? Promise.resolve() : pump();
          await new Promise((resolve) => setTimeout(resolve, 2000));
          const warmup = { steps: s.steps, elapsed: performance.now() - started };
          const startSteps = s.steps,
            startFrames = renderStats.frames;
          const samplesStart = performance.now();
          s.frameSamples.length = 0;
          const startCounts = Object.fromEntries(
            Object.entries(graphs.timings).map(([k, v]) => [k, v.count]),
          );
          const startTotals = structuredClone(graphs.timings);
          sampling = true;
          for (const metric of Object.values(graphs.timings)) {
            metric.samples.length = 0;
            if (metric.submitSamples) metric.submitSamples.length = 0;
            if (metric.readbackSamples) metric.readbackSamples.length = 0;
          }
          await new Promise((resolve) => setTimeout(resolve, 6000));
          driving = false;
          pumping = false;
          window.jelly!.pause();
          await pumpingDone;
          while (window.jelly!.busy) await new Promise((resolve) => setTimeout(resolve, 1));
          sampling = false;
          const elapsed = performance.now() - samplesStart;
          const steps = s.steps - startSteps,
            frames = renderStats.frames - startFrames;
          const result = {
            backend: 'WebGPU',
            baseline: graphs.baseline,
            paced,
            device: graphs.deviceLabel,
            fallback: graphs.fallbackAdapter,
            browser: navigator.userAgent,
            grid: s.mesh.grid,
            stiffness: s.settings.stiffness,
            dt: s.settings.dt,
            warmup,
            elapsed,
            steps,
            frames,
            updatesPerSecond: (steps * 1000) / elapsed,
            simulatedSeconds: steps * s.settings.dt,
            simulatedToWallRatio: (steps * s.settings.dt * 1000) / elapsed,
            renderFPS: (frames * 1000) / elapsed,
            stepMs: stepSamples,
            solverSamples,
            graphs: structuredClone(graphs.timings),
            means: Object.fromEntries(
              Object.entries(graphs.timings).map(([k, v]) => {
                const calls = v.count - startCounts[k];
                return [
                  k,
                  calls
                    ? {
                        runMs:
                          ((v.totalMs ?? 0) -
                            (startTotals[k as keyof typeof startTotals].totalMs ?? 0)) /
                          calls,
                        submitMs:
                          v.totalSubmitMs === undefined
                            ? null
                            : (v.totalSubmitMs -
                                (startTotals[k as keyof typeof startTotals].totalSubmitMs ?? 0)) /
                              calls,
                        readbackMs:
                          v.totalReadbackMs === undefined
                            ? null
                            : (v.totalReadbackMs -
                                (startTotals[k as keyof typeof startTotals].totalReadbackMs ?? 0)) /
                              calls,
                      }
                    : null,
                ];
              }),
            ),
            calls: Object.fromEntries(
              Object.entries(graphs.timings).map(([k, v]) => [k, v.count - startCounts[k]]),
            ),
            gpuCGSlots:
              (graphs.timings.batch.count - startCounts.batch) * 16 +
              (graphs.timings.step.count - startCounts.step) * 8 +
              (graphs.timings.cg.count - startCounts.cg) * 8,
            error: document.querySelector('#error')!.textContent,
          };
          const standaloneGraphMs: Record<'energy' | 'gradient' | 'hvp', number[]> = {
            energy: [],
            gradient: [],
            hvp: [],
          };
          const objective = graphs.objective(s.parameters(true));
          const direction = Float32Array.from(s.x, (_, i) => 0.1 * Math.cos(i * 1.7));
          for (const kind of ['energy', 'gradient', 'hvp'] as const) {
            for (let i = 0; i < 20; i++) {
              if (kind === 'energy') await objective.energy(s.x);
              else if (kind === 'gradient') await objective.evaluate(s.x);
              else await objective.hvp(s.x, direction);
              standaloneGraphMs[kind].push(graphs.timings[kind].last);
            }
          }
          return { ...result, standaloneGraphMs };
        },
        { paced },
      );
      expect(result.error).toBe('');
      expect(result.steps).toBeGreaterThan(0);
      results.push(result);
    } finally {
      await context.close();
    }
  }
  const path = testInfo.outputPath('performance.json');
  await writeFile(path, JSON.stringify(results, null, 2));
  await testInfo.attach('Sustained WebGPU performance', { path, contentType: 'application/json' });
  console.log(
    JSON.stringify(
      results.map((r) => ({
        baseline: r.baseline,
        paced: r.paced,
        updatesPerSecond: r.updatesPerSecond,
        renderFPS: r.renderFPS,
        graphMs: Object.fromEntries(Object.entries(r.means).map(([k, v]) => [k, v?.runMs ?? null])),
        stepMs: r.stepMs.reduce((a, b) => a + b, 0) / r.stepMs.length,
      })),
    ),
  );
  // Performance is reported on every device. Opt-in enforcement avoids treating
  // a software adapter or an arbitrary CI host as the measured hardware target.
  if (process.env.JELLY_PERF_REQUIRED === '1') {
    expect(results[1].fallback).toBe(false);
    expect(results[1].updatesPerSecond).toBeGreaterThanOrEqual(30);
    expect(results[1].renderFPS).toBeGreaterThanOrEqual(45);
    expect(results[2].updatesPerSecond).toBeGreaterThanOrEqual(40);
    expect(results[2].simulatedToWallRatio).toBeGreaterThanOrEqual(1);
  }
});
