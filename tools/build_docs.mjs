// Copyright (c) 2026 onnax contributors
// SPDX-License-Identifier: MIT
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const base = process.argv[2] ?? '/onnax';
assert(
  base === '' || /^\/[a-zA-Z0-9_-]+(?:\/[a-zA-Z0-9_-]+)*$/.test(base),
  `Invalid Pages base path ${JSON.stringify(base)}. Supply an empty string for a root site or a path such as /onnax without a trailing slash.`,
);
const demoBase = `${base}/jelly/`;
const dist = join(root, 'examples', 'jelly', 'dist');
const source = join(root, 'examples', 'jelly', 'public');
const read = (path) => {
  assert(
    existsSync(path),
    `Missing deployment artifact ${path}. Restore the checked-in public/models files, run npm ci in examples/jelly, then npm run build -- --base=${demoBase} before assembling docs.`,
  );
  const data = readFileSync(path);
  assert(
    data.length > 0,
    `Empty deployment artifact ${path}. Restore or rebuild this file before deploying.`,
  );
  return data;
};
const html = read(join(dist, 'index.html')).toString();
assert(
  html.includes(`src="${demoBase}assets/`) && html.includes(`href="${demoBase}assets/`),
  `Playground HTML does not target ${demoBase}. Rebuild with npm run build -- --base=${demoBase} in examples/jelly.`,
);
const manifest = JSON.parse(read(join(source, 'models', 'mesh.json')));
const models = [
  ...Object.keys(manifest.models).map((name) => `models/${name}.onnx`),
  ...Object.keys(manifest.runtime).map((name) => `models/runtime/${name}.onnx`),
  'models/mesh.json',
];
for (const name of models) {
  const original = read(join(source, name));
  const built = read(join(dist, name));
  assert(
    createHash('sha256').update(original).digest('hex') ===
      createHash('sha256').update(built).digest('hex'),
    `Deployed model ${name} differs from public/models. Rebuild the playground; do not mix generated model versions.`,
  );
}
const assets = readdirSync(join(dist, 'assets'));
const wasm = assets.filter((name) => /^ort-wasm-simd-threaded\.jsep-[\w-]+\.wasm$/.test(name));
assert(
  wasm.length === 1,
  'Expected one pinned ORT JSEP WASM host in dist/assets. Run npm ci and rebuild; do not substitute a different ORT runtime.',
);
assert(
  read(join(dist, 'assets', wasm[0]))
    .subarray(0, 4)
    .equals(Buffer.from([0, 97, 115, 109])),
  `Invalid WASM header in ${wasm[0]}. Run npm ci and rebuild the playground.`,
);
for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
  if (match[1].startsWith(demoBase)) read(join(dist, match[1].slice(demoBase.length)));
}
const output = join(root, '_site');
rmSync(output, { recursive: true, force: true });
mkdirSync(output);
cpSync(join(root, 'docs'), output, { recursive: true });
cpSync(dist, join(output, 'jelly'), { recursive: true });
writeFileSync(join(output, '.nojekyll'), '');
console.log(
  `Verified ${models.length} model/manifest files and matching ORT WASM assets. Docs: ${base || '/'}, playground: ${demoBase}; output: ${output}`,
);
