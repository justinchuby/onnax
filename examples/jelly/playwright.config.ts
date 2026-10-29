import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.browser.ts',
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:5175',
    browserName: 'chromium',
    channel: process.env.JELLY_BROWSER ?? (process.platform === 'win32' ? 'msedge' : 'chromium'),
    headless: true,
    screenshot: 'only-on-failure',
    launchOptions: { args: ['--enable-unsafe-webgpu'] },
  },
  webServer: {
    command: `npm run ${process.env.JELLY_PREVIEW === '1' ? 'preview' : 'dev'} -- --port 5175 --strictPort`,
    url: 'http://127.0.0.1:5175',
    reuseExistingServer: !process.env.CI,
  },
});
