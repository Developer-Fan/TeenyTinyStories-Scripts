// Step 1 environment probe. One logged call. Writes step1_probe_log.jsonl in this directory.
// Loads Puter.js via the SDK already installed in cloud_training/eval/node_modules (read-only use).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const EVAL_DIR = "/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval";
const OUT_DIR = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(EVAL_DIR, "package.json"));
const { init } = require("@heyputer/puter.js/src/init.cjs");

const MODEL = "infron:openai/gpt-5.6-luna"; // identifier taken from the earlier runner; NOT verified
const TOKEN = process.env.PUTER_AUTH_TOKEN || undefined;
const messages = [{ role: "user", content: "Reply with exactly the single word: OK" }];
const logPath = path.join(OUT_DIR, "step1_probe_log.jsonl");

const puter = init(TOKEN);
const endpoint = {
  PUTER_API_ORIGIN: globalThis.PUTER_API_ORIGIN ?? null,
  puter_APIOrigin: puter.APIOrigin ?? null,
};
const rec = {
  step: "step1_probe",
  utc_timestamp: new Date().toISOString(),
  token_env_present: Boolean(TOKEN),
  endpoint,
  model_requested: MODEL,
  temperature_sent: 0,
  prompt_sent: messages,
};
const t0 = Date.now();
try {
  const r = await Promise.race([
    puter.ai.chat(messages, { model: MODEL, temperature: 0 }),
    new Promise((_, rej) => setTimeout(() => rej(new Error("T90 local timeout")), 90000)),
  ]);
  rec.ok = true;
  rec.raw_response = r;
} catch (e) {
  rec.ok = false;
  rec.error = { name: e?.name, message: String(e?.message ?? e), code: e?.code ?? null,
                status: e?.status ?? e?.response?.status ?? null,
                raw: (() => { try { return JSON.parse(JSON.stringify(e)); } catch { return null; } })() };
}
rec.latency_ms = Date.now() - t0;
fs.appendFileSync(logPath, JSON.stringify(rec) + "\n");
console.log(JSON.stringify({ ok: rec.ok, model_requested: MODEL, endpoint, latency_ms: rec.latency_ms,
  error_message: rec.error?.message ?? null, error_status: rec.error?.status ?? null }, null, 2));
process.exit(0);
