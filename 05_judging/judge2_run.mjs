// Second-judge runner. Exactly ONE API attempt per input line. No retries inside this script.
// Retries are a separate, explicit pass (attempt=2 inputs) and are logged as their own records.
// Usage: PUTER_AUTH_TOKEN=... node judge2_run.mjs <attempts.jsonl> <calls_log.jsonl> <temperature>
// Input line:  {"key","attempt","phase","tag","messages"}
// Output line (appended to calls_log.jsonl): full record of the call, including the raw response.
// Resume rule: an input whose (key, attempt) already appears in calls_log.jsonl is skipped (no new call).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const EVAL_DIR = "/Users/yanfu/Documents/TeenyTinyStories/cloud_training/eval";
const require = createRequire(path.join(EVAL_DIR, "package.json"));
const { init } = require("@heyputer/puter.js/src/init.cjs");

const [attemptsPath, logPath, tempArg] = process.argv.slice(2);
const MODEL = "infron:openai/gpt-5.6-luna";   // requested identifier (unverified; the API response is what gets logged)
const TEMP = Number(tempArg);
if (!Number.isFinite(TEMP)) throw new Error("temperature argument missing or not a number");
const TOKEN = process.env.PUTER_AUTH_TOKEN;
if (!TOKEN) throw new Error("PUTER_AUTH_TOKEN not set in environment");

const puter = init(TOKEN);
const done = new Set();
if (fs.existsSync(logPath)) {
  for (const l of fs.readFileSync(logPath, "utf8").split("\n")) {
    if (!l.trim()) continue;
    const r = JSON.parse(l);
    done.add(`${r.key}#${r.attempt}`);
  }
}
const items = fs.readFileSync(attemptsPath, "utf8").split("\n").filter(Boolean)
  .map((l) => JSON.parse(l))
  .filter((it) => !done.has(`${it.key}#${it.attempt}`));
console.log(`temperature=${TEMP} model_requested=${MODEL} todo=${items.length} already_logged_skipped=${done.size ? "see log" : 0}`);

for (const it of items) {
  const rec = {
    key: it.key,
    attempt: it.attempt,
    phase: it.phase,
    tag: it.tag,
    utc_timestamp: new Date().toISOString(),
    endpoint: puter.APIOrigin ?? null,
    model_requested: MODEL,
    temperature_sent: TEMP,
    prompt_sent: it.messages,
  };
  const t0 = Date.now();
  try {
    const r = await Promise.race([
      puter.ai.chat(it.messages, { model: MODEL, temperature: TEMP }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("local timeout 90s")), 90000)),
    ]);
    const raw = JSON.parse(JSON.stringify(r));
    rec.ok = true;
    rec.raw_response = raw;
    rec.response_text = typeof raw?.message?.content === "string" ? raw.message.content : null;
    rec.finish_reason = raw?.finish_reason ?? null;
    rec.model_returned = raw?.model ?? raw?.message?.model ?? null;  // null = API did not report a model
    rec.error = "";
    rec.error_status = null;
  } catch (e) {
    rec.ok = false;
    rec.raw_response = null;
    rec.response_text = null;
    rec.finish_reason = null;
    rec.model_returned = null;
    rec.error = String(e?.message ?? e);
    rec.error_status = e?.status ?? null;
  }
  rec.latency_ms = Date.now() - t0;
  fs.appendFileSync(logPath, JSON.stringify(rec) + "\n");
  console.log(`${it.phase} ${it.key}#${it.attempt} ok=${rec.ok} finish=${rec.finish_reason} ${rec.latency_ms}ms${rec.error ? " ERR=" + rec.error : ""}`);
}
console.log("done");
process.exit(0);
