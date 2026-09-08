/**
 * The mandatory env block: every sweep record carries the environment that
 * produced it (headless desktop Chromium is not a user's phone).
 */
import { createRequire } from 'node:module';
import * as os from 'node:os';

import type { AdapterInfo } from '../shared/protocol.ts';

const require = createRequire(import.meta.url);

export function litertCoreVersion(): string {
  const pkg = require('@litertjs/core/package.json') as { version: string };
  return pkg.version;
}

function osName(): string {
  switch (process.platform) {
    case 'darwin':
      return 'macOS';
    case 'linux':
      return 'Linux';
    case 'win32':
      return 'Windows';
    default:
      return process.platform;
  }
}

export interface SweepEnv {
  browser: string;
  browser_version: string;
  headless: boolean;
  os: string;
  os_version: string;
  jspi: boolean;
  webgpu_adapter: AdapterInfo | null;
  litertjs_core_version: string;
  machine_label: string;
}

export function buildEnv(params: {
  browserVersion: string;
  headless: boolean;
  jspi: boolean;
  adapter: AdapterInfo | null;
  machineLabel: string;
}): SweepEnv {
  return {
    browser: 'chromium',
    browser_version: params.browserVersion,
    headless: params.headless,
    os: osName(),
    os_version: os.release(),
    jspi: params.jspi,
    webgpu_adapter: params.adapter,
    litertjs_core_version: litertCoreVersion(),
    machine_label: params.machineLabel,
  };
}
