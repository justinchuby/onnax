// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import type { Simulation } from './simulation';

export function forceArrow(fx: number, fy: number) {
  const magnitude = Math.hypot(fx, fy);
  if (magnitude < 0.02) return null;
  const strength = Math.tanh(magnitude / 6);
  return { dx: fx / magnitude, dy: fy / magnitude, length: 6 + 30 * strength, strength };
}

export class Renderer {
  private readonly context: CanvasRenderingContext2D;
  width = 1;
  height = 1;
  scale = 1;
  private readonly observer: ResizeObserver;
  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly simulation: Simulation,
  ) {
    const context = canvas.getContext('2d');
    if (!context)
      throw new Error('Canvas 2D is unavailable. Enable browser graphics acceleration and reload.');
    this.context = context;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
  }
  private resize(): void {
    const box = this.canvas.getBoundingClientRect();
    this.width = box.width;
    this.height = box.height;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(box.width * ratio);
    this.canvas.height = Math.round(box.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.scale = Math.min(this.width / 2.8, (this.height - 120) / 2.5);
    this.draw();
  }
  screen(x: number, y: number): [number, number] {
    return [this.width / 2 + x * this.scale, this.height / 2 + y * this.scale];
  }
  world(clientX: number, clientY: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [
      (clientX - rect.left - this.width / 2) / this.scale,
      (clientY - rect.top - this.height / 2) / this.scale,
    ];
  }
  nearest(point: [number, number]): number | null {
    let best = -1,
      distance = 0.22;
    for (let node = 0; node < this.simulation.mesh.positions.length; node++) {
      if (this.simulation.pinned.has(node)) continue;
      const d = Math.hypot(
        this.simulation.x[2 * node] - point[0],
        this.simulation.x[2 * node + 1] - point[1],
      );
      if (d < distance) {
        distance = d;
        best = node;
      }
    }
    return best === -1 ? null : best;
  }
  draw(): void {
    const c = this.context,
      s = this.simulation;
    c.clearRect(0, 0, this.width, this.height);
    const glow = c.createRadialGradient(
      this.width / 2,
      this.height * 0.47,
      5,
      this.width / 2,
      this.height / 2,
      this.width * 0.65,
    );
    glow.addColorStop(0, '#342641');
    glow.addColorStop(0.55, '#211c34');
    glow.addColorStop(1, '#16162b');
    c.fillStyle = glow;
    c.fillRect(0, 0, this.width, this.height);
    c.fillStyle = '#6f608733';
    for (let x = (this.width / 2) % 24; x < this.width; x += 24) {
      for (let y = (this.height / 2) % 24; y < this.height; y += 24) {
        c.beginPath();
        c.arc(x, y, 0.7, 0, 2 * Math.PI);
        c.fill();
      }
    }
    const p = (node: number): [number, number] => this.screen(s.x[2 * node], s.x[2 * node + 1]);
    const n = s.mesh.grid;
    // Ghost rest shape provides a fixed reference for deformation.
    c.setLineDash([3, 5]);
    c.strokeStyle = '#9b83b52a';
    c.lineWidth = 1;
    for (const [a, b] of s.mesh.edges) {
      c.beginPath();
      c.moveTo(...this.screen(s.rest[a * 2], s.rest[a * 2 + 1]));
      c.lineTo(...this.screen(s.rest[b * 2], s.rest[b * 2 + 1]));
      c.stroke();
    }
    c.setLineDash([]);
    for (let row = 0; row < n - 1; row++) {
      for (let col = 0; col < n - 1; col++) {
        const a = row * n + col;
        const triangles = [
          [a, a + 1, a + n + 1],
          [a, a + n + 1, a + n],
        ];
        for (const triangle of triangles) {
          let strain = 0;
          for (let edge = 0; edge < 3; edge++) {
            const u = triangle[edge],
              v = triangle[(edge + 1) % 3];
            const rest = Math.hypot(
              s.rest[2 * u] - s.rest[2 * v],
              s.rest[2 * u + 1] - s.rest[2 * v + 1],
            );
            strain +=
              Math.abs(
                Math.hypot(s.x[2 * u] - s.x[2 * v], s.x[2 * u + 1] - s.x[2 * v + 1]) / rest - 1,
              ) / 3;
          }
          const tint = s.settings.overlay === 'strain' ? Math.min(strain * 2, 1) : 0;
          const hue = 285 + 48 * tint + row * 2;
          c.fillStyle = `hsla(${hue},${50 + tint * 25}%,${66 + col * 2}%,0.64)`;
          c.strokeStyle = `hsla(${hue},65%,83%,0.07)`;
          c.beginPath();
          c.moveTo(...p(triangle[0]));
          for (const node of triangle.slice(1)) c.lineTo(...p(node));
          c.closePath();
          c.fill();
          c.stroke();
        }
      }
    }
    for (const [a, b] of s.mesh.edges) {
      const diagonal =
        Math.abs((a % n) - (b % n)) === 1 && Math.abs(Math.floor(a / n) - Math.floor(b / n)) === 1;
      c.lineWidth = diagonal ? 0.65 : 1;
      c.strokeStyle = diagonal ? '#eecaf329' : '#f6d5f780';
      c.beginPath();
      c.moveTo(...p(a));
      c.lineTo(...p(b));
      c.stroke();
    }
    const boundary = [
      ...Array.from({ length: n }, (_, i) => i),
      ...Array.from({ length: n - 1 }, (_, i) => (i + 1) * n + n - 1),
      ...Array.from({ length: n - 1 }, (_, i) => n * n - 2 - i),
      ...Array.from({ length: n - 2 }, (_, i) => (n - 2 - i) * n),
    ];
    c.beginPath();
    boundary.forEach((node, i) => {
      if (i) c.lineTo(...p(node));
      else c.moveTo(...p(node));
    });
    c.closePath();
    c.lineWidth = 1.8;
    c.strokeStyle = '#f6ceee';
    c.shadowColor = '#efa5ec';
    c.shadowBlur = 12;
    c.stroke();
    c.shadowBlur = 0;
    if (s.settings.overlay === 'force') {
      for (let node = 0; node < s.mesh.positions.length; node++) {
        if (s.pinned.has(node)) continue;
        const arrow = forceArrow(s.forces[node * 2], s.forces[node * 2 + 1]);
        if (!arrow) continue;
        const [x, y] = p(node);
        const { dx, dy, length, strength } = arrow;
        c.strokeStyle = `hsl(${160 - 70 * strength}, 85%, 78%)`;
        c.lineWidth = 1.6;
        c.beginPath();
        c.moveTo(x, y);
        c.lineTo(x + length * dx, y + length * dy);
        c.lineTo(x + (length - 4) * dx + 2.5 * dy, y + (length - 4) * dy - 2.5 * dx);
        c.moveTo(x + length * dx, y + length * dy);
        c.lineTo(x + (length - 4) * dx - 2.5 * dy, y + (length - 4) * dy + 2.5 * dx);
        c.stroke();
      }
    }
    for (let node = 0; node < s.mesh.positions.length; node++) {
      const pinned = s.pinned.has(node),
        active = s.dragging?.node === node;
      c.fillStyle = pinned ? '#f6d699' : active ? '#fff7ea' : '#f8d7f0cc';
      c.beginPath();
      c.arc(...p(node), pinned ? 4 : active ? 5 : 2, 0, Math.PI * 2);
      c.fill();
      if (pinned || active) {
        c.strokeStyle = pinned ? '#eccc9655' : '#f8dcebbb';
        c.lineWidth = 1;
        c.beginPath();
        c.arc(...p(node), pinned ? 9 : 12, 0, Math.PI * 2);
        c.stroke();
      }
    }
    if (s.dragging) {
      const [x, y] = this.screen(...s.dragging.position);
      c.setLineDash([3, 4]);
      c.strokeStyle = '#fff2d6aa';
      c.beginPath();
      c.moveTo(...p(s.dragging.node));
      c.lineTo(x, y);
      c.stroke();
      c.setLineDash([]);
      c.beginPath();
      c.arc(x, y, 6, 0, Math.PI * 2);
      c.stroke();
    }
    c.fillStyle = '#847395';
    c.font = '9px Consolas, monospace';
    c.textAlign = 'left';
    c.fillText(
      s.settings.overlay === 'force'
        ? 'ONNX FORCE · mint → gold · nonlinear scale'
        : s.settings.overlay === 'strain'
          ? 'STRAIN · lavender → rose'
          : 'REST SHAPE · dashed',
      25,
      this.height - 47,
    );
  }
}
