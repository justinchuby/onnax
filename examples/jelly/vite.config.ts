import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { defineConfig } from 'vite';

const hash = createHash('sha256');
for (const name of [
  'mesh.json',
  ...['energy', 'gradient', 'hvp'].flatMap((kind) => [
    `${kind}.onnx`,
    `${kind}.svg`,
    `runtime/${kind}.svg`,
  ]),
  ...['energy', 'gradient', 'hvp', 'cg', 'step', 'batch'].map((kind) => `runtime/${kind}.onnx`),
]) {
  const path = new URL(`./public/models/${name}`, import.meta.url);
  if (!existsSync(path))
    throw new Error(
      `Cannot build jelly: missing model artifact ${path.pathname}. Restore public/models from Git or regenerate the complete set with python examples/jelly/export_models.py from the repository root.`,
    );
  hash.update(name).update(readFileSync(path));
}

export default defineConfig({
  define: { __JELLY_MODEL_VERSION__: JSON.stringify(hash.digest('hex')) },
});
