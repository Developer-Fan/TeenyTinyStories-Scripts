// Cached judge batch runner: node judge_batch.mjs <jobs.jsonl> <out.jsonl> [concurrency]
// Each job line: {"key": "...", "messages": [{role, content}...], "tag": {...}}
// Appends {"key", "tag", "response_text", "model_id", "latency_ms", "attempts", "error"}
// Skips keys already present in out file (reruns cost nothing).
import { init } from "@heyputer/puter.js/src/init.cjs";
import fs from "node:fs";

const MODEL = "infron:openai/gpt-5.6-luna";
const [jobsPath, outPath, concArg] = process.argv.slice(2);
const CONC = parseInt(concArg || "2", 10);

const puter = init(process.env.PUTER_AUTH_TOKEN);
const done = new Set();
if (fs.existsSync(outPath)) {
  for (const line of fs.readFileSync(outPath, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try { done.add(JSON.parse(line).key); } catch {}
  }
}
const jobs = fs.readFileSync(jobsPath, "utf8").split("\n").filter(Boolean)
  .map((l) => JSON.parse(l)).filter((j) => !done.has(j.key));
console.log(`cached=${done.size} todo=${jobs.length} conc=${CONC}`);

function textOf(r) {
  if (!r) return "";
  if (typeof r === "string") return r;
  const c = r.message?.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((p) => p.text ?? "").join("");
  return r.text ?? JSON.stringify(r).slice(0, 500);
}

async function one(job) {
  let attempts = 0, lastErr = "";
  while (attempts < 4) {
    attempts++;
    const t0 = Date.now();
    try {
      const r = await Promise.race([
        puter.ai.chat(job.messages, { model: MODEL, temperature: 0 }),
        new Promise((_, rej) => setTimeout(() => rej(new Error("T90")), 90000)),
      ]);
      return { key: job.key, tag: job.tag, response_text: textOf(r),
               model_id: MODEL, latency_ms: Date.now() - t0, attempts, error: "" };
    } catch (e) {
      lastErr = String((e && e.message) || e).slice(0, 200);
      await new Promise((r) => setTimeout(r, 2000 * attempts));
    }
  }
  return { key: job.key, tag: job.tag, response_text: "", model_id: MODEL,
           latency_ms: 0, attempts, error: lastErr };
}

const out = fs.createWriteStream(outPath, { flags: "a" });
let i = 0, errs = 0;
async function worker() {
  while (i < jobs.length) {
    const job = jobs[i++];
    const res = await one(job);
    if (res.error) errs++;
    out.write(JSON.stringify(res) + "\n");
    if (i % 50 === 0) console.log(`  ${i}/${jobs.length} errs=${errs}`);
  }
}
await Promise.all(Array.from({ length: Math.min(CONC, jobs.length) }, worker));
out.close();
console.log(`done errs=${errs}`);
process.exit(0);
