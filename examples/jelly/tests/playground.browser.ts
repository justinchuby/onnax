// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import { expect, test } from '@playwright/test';
import type {} from '../src/main';

declare global {
  interface Window {
    gpuDispatches: number;
    gpuSubmissions: number;
  }
}

test('GPU-only autodiff, force field, stiff solve, controls, pointer/touch and responsive rendering', async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('requestfailed', (request) =>
    errors.push(`${request.url()}: ${request.failure()?.errorText}`),
  );
  page.on('response', (response) => {
    if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`);
  });
  await page.addInitScript(() => {
    window.gpuDispatches = 0;
    window.gpuSubmissions = 0;
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function (...args) {
      window.gpuSubmissions++;
      return submit.apply(this, args);
    };
    const dispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      window.gpuDispatches++;
      return dispatch.apply(this, args);
    };
  });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto('./');
  await page.waitForFunction(
    () => window.jelly || !document.querySelector<HTMLElement>('#error')!.hidden,
    undefined,
    { timeout: 60_000 },
  );
  await expect(page.locator('#error')).toBeHidden();
  await expect(page.locator('#pause')).toBeEnabled();
  await page.evaluate(() => window.jelly!.pause());
  await page.waitForFunction(() => !window.jelly!.busy);
  await expect(page.locator('#backend-badge')).toContainText('WEBGPU');
  await expect(page.locator('#device-info')).toContainText('ONNX opset 22');
  await expect(page.locator('#explicit')).toHaveCount(0);
  await expect(page.locator('#overlay')).toHaveValue('force');
  await page.locator('#computation summary').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#model-graphs')).toBeVisible();
  await expect(page.locator('#model-graphs section')).toHaveCount(3);
  await expect(page.locator('#model-graphs img')).toHaveCount(6);
  await expect(page.locator('#model-graphs')).toContainText('ONNX opset 22');
  await expect(page.locator('#model-graphs pre')).toHaveCount(0);
  for (const link of await page.locator('#model-graphs a').all()) {
    const url = new URL((await link.getAttribute('href'))!);
    expect(url.origin).toBe('https://netron.app');
    expect(url.searchParams.get('url')).toMatch(
      /^https:\/\/www\.justinchuby\.com\/onnax\/jelly\/models\/(runtime\/)?(energy|gradient|hvp)\.onnx$/,
    );
    await expect(link).toHaveAttribute('target', '_blank');
  }
  for (const image of await page.locator('#model-graphs img').all()) {
    await image.scrollIntoViewIfNeeded();
    await expect(image).toHaveJSProperty('complete', true);
    expect(
      await image.evaluate((element) => (element as HTMLImageElement).naturalWidth),
    ).toBeGreaterThan(0);
  }
  await page.keyboard.press('Enter');
  await expect(page.locator('#error')).toBeHidden();

  const derivatives = await page.evaluate(async () => {
    const { simulation: s, graphs } = window.jelly!;
    s.reset();
    const restPhysical = await graphs.objective(s.parameters(false)).evaluate(s.rest);
    for (let node = 0; node < s.mesh.positions.length; node++) {
      if (Math.abs(restPhysical.gradient[2 * node]) > 1e-4)
        throw new Error(
          `Rest force x at node ${node} is not zero; inspect the physical gradient graph.`,
        );
      const expected = s.pinned.has(node) ? 0 : -s.mesh.mass * 9.81;
      if (Math.abs(restPhysical.gradient[2 * node + 1] - expected) > 1e-4)
        throw new Error(
          `Rest gravity gradient at node ${node} differs from ${expected}; inspect force sign and pin projection.`,
        );
    }
    s.x[42] += 0.045;
    s.x[43] -= 0.03;
    const objective = graphs.objective(s.parameters(true));
    const x = s.x.slice();
    const v = Float32Array.from(x, (_, i) => Math.sin(i * 1.7) * 0.25);
    const before = window.gpuSubmissions;
    await objective.energy(x);
    const afterEnergy = window.gpuSubmissions;
    const center = await objective.evaluate(x);
    const afterGradient = window.gpuSubmissions;
    const hv = await objective.hvp(x, v);
    const afterHvp = window.gpuSubmissions;
    const h = 0.001;
    const plus = Float32Array.from(x, (a, i) => a + h * v[i]);
    const minus = Float32Array.from(x, (a, i) => a - h * v[i]);
    const p = await objective.evaluate(plus),
      m = await objective.evaluate(minus);
    const directionalGradient = center.gradient.reduce((sum, a, i) => sum + a * v[i], 0);
    const directionalFD = (p.energy - m.energy) / (2 * h);
    const hvFD = Float32Array.from(hv, (_, i) => (p.gradient[i] - m.gradient[i]) / (2 * h));
    const hvError = Math.hypot(...hv.map((a, i) => a - hvFD[i])) / Math.hypot(...hv);
    s.reset();
    return {
      directionalGradient,
      directionalFD,
      hvError,
      submissions: [afterEnergy - before, afterGradient - afterEnergy, afterHvp - afterGradient],
      recordedComputeDispatches: window.gpuDispatches,
    };
  });
  for (const count of derivatives.submissions) expect(count).toBeGreaterThan(0);
  expect(derivatives.recordedComputeDispatches).toBeGreaterThan(0);
  expect(Math.abs(derivatives.directionalFD - derivatives.directionalGradient)).toBeLessThan(0.01);
  expect(derivatives.hvError).toBeLessThan(0.005);

  await page.locator('#damping').focus();
  await page.keyboard.press('End');
  await expect(page.locator('#damping-value')).toContainText('12.0');
  await page.locator('#damping').focus();
  await page.keyboard.press('Home');
  await expect(page.locator('#damping-value')).toContainText('0.0');
  await page.locator('#gravity').uncheck();
  await page.locator('#pins').selectOption('corners');
  await expect.poll(() => page.evaluate(() => window.jelly!.simulation.pinned.size)).toBe(4);
  await page.locator('#pins').selectOption('none');
  await expect.poll(() => page.evaluate(() => window.jelly!.simulation.pinned.size)).toBe(0);
  await page.locator('#pins').selectOption('top');
  await page.locator('#gravity').check();

  // Real pointer events acquire a free node; the graph tether deforms the sheet.
  await page.locator('#reset').click();
  const point = await page.evaluate(() => {
    const { renderer: r, simulation: s } = window.jelly!;
    const rect = r.canvas.getBoundingClientRect();
    const p = r.screen(s.x[42], s.x[43]);
    return { x: rect.left + p[0], y: rect.top + p[1] };
  });
  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.move(point.x + 45, point.y - 35, { steps: 4 });
  expect(await page.evaluate(() => window.jelly!.simulation.dragging?.node)).toBe(21);
  await page.evaluate(async () => {
    for (let i = 0; i < 3; i++) await window.jelly!.step();
  });
  expect(
    await page.evaluate(() => window.jelly!.simulation.x[42] - window.jelly!.simulation.rest[42]),
  ).toBeGreaterThan(0.015);
  const physicalForceError = await page.evaluate(async () => {
    const { simulation: s, graphs } = window.jelly!;
    const physical = await graphs.objective(s.parameters(false)).evaluate(s.x);
    return Math.max(...physical.gradient.map((value, i) => Math.abs(s.forces[i] + value)));
  });
  expect(physicalForceError).toBeLessThan(0.005);
  await expect(page.locator('#force-max')).toContainText('units');
  await page.mouse.up();
  expect(await page.evaluate(() => window.jelly!.simulation.dragging)).toBeNull();
  await page.locator('#overlay').selectOption('force');
  await expect
    .poll(() => page.evaluate(() => window.jelly!.simulation.settings.overlay))
    .toBe('force');
  await page.locator('#overlay').selectOption('none');
  await page.locator('#overlay').selectOption('force');

  // Pause/reset are UI actions, not hidden state changes.
  const pausedSteps = await page.evaluate(() => window.jelly!.simulation.steps);
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  expect(await page.evaluate(() => window.jelly!.simulation.steps)).toBe(pausedSteps);
  await page.locator('#pause').click();
  await expect
    .poll(() => page.evaluate(() => window.jelly!.simulation.steps))
    .toBeGreaterThan(pausedSteps);
  await page.locator('#pause').click();
  await page.waitForFunction(() => !window.jelly!.busy);
  await page.locator('#reset').click();
  expect(await page.evaluate(() => window.jelly!.simulation.steps)).toBe(0);

  await page.locator('#stiff-preset').click();
  await page.evaluate(() => window.jelly!.pause());
  await page.waitForFunction(() => !window.jelly!.busy);
  await expect(page.locator('#stiffness-value')).toHaveText('8000');
  await expect(page.locator('#timestep-value')).toHaveText('40 ms');
  const stiff = await page.evaluate(async () => {
    const { simulation: s, graphs } = window.jelly!;
    s.reset();
    s.frameSamples.length = 0;
    const solves = [];
    for (let i = 0; i < 15; i++) {
      await window.jelly!.step();
      if (s.lastSolve)
        solves.push({
          initialResidual: s.lastSolve.initialResidual,
          residual: s.lastSolve.residual,
          initialEnergy: s.lastSolve.initialEnergy,
          energy: s.lastSolve.energy,
          converged: s.lastSolve.converged,
          termination: s.lastSolve.termination,
        });
    }
    return {
      x: Array.from(s.x),
      steps: s.steps,
      pins: [s.x[0], s.x[1], s.x[10], s.x[11]],
      solves,
      stepTimes: s.frameSamples.slice(),
      device: graphs.deviceLabel,
      fallbackAdapter: graphs.fallbackAdapter,
      userAgent: navigator.userAgent,
      graphTimings: Object.fromEntries(
        Object.entries(graphs.timings).map(([name, v]) => [name, v.samples.slice(-30)]),
      ),
      dispatches: window.gpuDispatches,
    };
  });
  await expect(page.locator('#error')).toBeHidden();
  expect(stiff.solves).toHaveLength(15);
  expect(stiff.steps).toBe(15);
  for (const solve of stiff.solves) {
    expect(solve.energy).toBeLessThanOrEqual(solve.initialEnergy + 1e-5);
    if (solve.initialResidual > 0.05) expect(solve.residual).toBeLessThan(solve.initialResidual);
    expect(solve.residual).toBeLessThan(0.03);
    if (!solve.converged) expect(['precision', 'budget']).toContain(solve.termination);
  }
  expect(stiff.solves[0].residual).toBeLessThan(stiff.solves[0].initialResidual * 0.05);
  expect(stiff.x.every(Number.isFinite)).toBe(true);
  expect(Math.max(...stiff.x.map(Math.abs))).toBeLessThan(2);
  expect(stiff.pins).toEqual([
    Math.fround(-0.65),
    Math.fround(-0.65),
    Math.fround(0.65),
    Math.fround(-0.65),
  ]);
  await page.locator('#stiffness').focus();
  await page.keyboard.press('Home');
  for (let i = 0; i < 40; i++) await page.keyboard.press('ArrowRight');
  await page.locator('#timestep').focus();
  await page.keyboard.press('Home');
  for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowRight');
  await page.locator('#reset').click();
  await page.locator('#tug').click();
  await expect
    .poll(() => page.evaluate(() => window.jelly!.simulation.steps))
    .toBeGreaterThanOrEqual(8);
  await page.evaluate(() => window.jelly!.pause());
  await page.waitForFunction(() => !window.jelly!.busy);
  await expect(page.locator('#error')).toBeHidden();
  const screenshot = testInfo.outputPath('jelly-desktop.png');
  await page.screenshot({ path: screenshot, fullPage: true });
  await testInfo.attach('Desktop playground', { path: screenshot, contentType: 'image/png' });
  await testInfo.attach('WebGPU measurements', {
    body: JSON.stringify({ derivatives, stiff, physicalForceError }, null, 2),
    contentType: 'application/json',
  });
  console.log(
    JSON.stringify({
      device: stiff.device,
      fallback: stiff.fallbackAdapter,
      stiffStepMs: stiff.stepTimes,
      graphQueueSubmissions: derivatives.submissions,
      convergedStiffSteps: stiff.solves.filter((s) => s.converged).length,
    }),
  );

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#scene').scrollIntoViewIfNeeded();
  const touchPoint = await page.evaluate(() => {
    const { renderer: r, simulation: s } = window.jelly!;
    const rect = r.canvas.getBoundingClientRect(),
      p = r.screen(s.x[42], s.x[43]);
    return { x: rect.left + p[0], y: rect.top + p[1] };
  });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Input.dispatchTouchEvent', {
    type: 'touchStart',
    touchPoints: [{ x: touchPoint.x, y: touchPoint.y }],
  });
  await cdp.send('Input.dispatchTouchEvent', {
    type: 'touchMove',
    touchPoints: [{ x: touchPoint.x + 20, y: touchPoint.y - 15 }],
  });
  expect(await page.evaluate(() => window.jelly!.simulation.dragging?.node)).toBe(21);
  await page.evaluate(() => window.jelly!.step());
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  expect(await page.evaluate(() => window.jelly!.simulation.dragging)).toBeNull();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.screenshot({ path: testInfo.outputPath('jelly-mobile.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('unsupported browser reports an actionable error, never a fake fallback', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('./');
  await expect(page.locator('#error')).toContainText('WebGPU is unavailable');
  await expect(page.locator('#error')).toContainText('No WASM physics fallback');
  await expect(page.locator('#pause')).toBeDisabled();
});
