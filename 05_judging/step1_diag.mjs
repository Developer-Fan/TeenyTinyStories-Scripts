// Step 1 diagnostic: one logged call with an explicitly given model string.
// Usage: PUTER_AUTH_TOKEN=... node step1_diag.mjs <model-string> <label>
// Appends a record to step1_diag_log.jsonl in this directory. Token is read from env only, never written.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const EVAL_DIR = "/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval";
const OUT_DIR = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(EVAL_DIR, "package.json"));
const { init } = require("@heyputer/puter.js/src/init.cjs");

const [MODEL, LABEL] = process.argv.slice(2);
const TOKEN = process.env.PUTER_AUTH_TOKEN || undefined;
const messages = [{ role: "user", content: "Reply with exactly the single word: OK" }];
const logPath = path.join(OUT_DIR, "step1_diag_log.jsonl");

const puter = init(TOKEN);
const rec = {
  step: "step1_diag",
  label: LABEL,
  utc_timestamp: new Date().toISOString(),
  token_env_present: Boolean(TOKEN),
  endpoint: { puter_APIOrigin: puter.APIOrigin ?? null },
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
  rec.raw_response = JSON.parse(JSON.stringify(r));
  rec.response_top_level_keys = Object.keys(rec.raw_response);
  rec.response_has_model_field = "model" in rec.raw_response || (rec.raw_response.message && "model" in rec.raw_response.message);
} catch (e) {
  rec.ok = false;
  rec.error = { message: String(e?.message ?? e), code: e?.code ?? null, status: e?.status ?? null,
                raw: (() => { try { return JSON.parse(JSON.stringify(e)); } catch { return null; } })() };
}
rec.latency_ms = Date.now() - t0;
fs.appendFileSync(logPath, JSON.stringify(rec) + "\n");
console.log(JSON.stringify({ label: LABEL, ok: rec.ok, model_requested: MODEL,
  response_top_level_keys: rec.response_top_level_keys ?? null,
  response_has_model_field: rec.response_has_model_field ?? null,
  error_message: rec.error?.message ?? null, error_status: rec.error?.status ?? null,
  latency_ms: rec.latency_ms }, null, 2));
process.exit(0);
