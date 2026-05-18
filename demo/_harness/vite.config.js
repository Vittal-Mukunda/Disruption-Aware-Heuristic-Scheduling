import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";
import { execFile } from "child_process";

// Test harness for demo/dahs_dashboard.jsx (which lives two levels up).
// The artifact is outside this dir, so bare imports it makes (react, recharts,
// lucide-react) are aliased explicitly to this harness's node_modules.
const here = path.dirname(fileURLToPath(import.meta.url));
const nm = path.join(here, "node_modules");
const repoRoot = path.join(here, "..", "..");

const ALLOWED_BASELINES = new Set([
  "fifo", "fefo", "wspt", "atc", "greedy_mpc", "snapshot_xgb", "linucb", "ppo_fair",
]);

// ---------------------------------------------------------------------------
// Dev-only endpoint:  GET /api/run?seed=<int>&baseline=<key>
// Runs the REAL demo/build_run_log.py harness (which drives the actual
// WarehouseEnv + DAHS + baseline policies) and streams back its JSON log.
// Nothing here fabricates data -- the dashboard replays exactly what the
// Python simulator produced. Results are cached per (seed, baseline) so
// re-selecting a run during a talk is instant.
// ---------------------------------------------------------------------------
function dahsRunnerPlugin() {
  const cache = new Map();
  return {
    name: "dahs-runner",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/api/run")) return next();
        const url = new URL(req.url, "http://localhost");
        const seedRaw = url.searchParams.get("seed") ?? "42";
        const baseline = url.searchParams.get("baseline") ?? "fifo";
        const fresh = url.searchParams.get("fresh") === "1";
        const seed = Number.parseInt(seedRaw, 10);

        const fail = (code, msg) => {
          res.statusCode = code;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: msg }));
        };
        if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
          return fail(400, `Invalid seed "${seedRaw}" - enter an integer in 0..4294967295.`);
        if (!ALLOWED_BASELINES.has(baseline))
          return fail(400, `Unknown baseline "${baseline}".`);

        const key = `${seed}|${baseline}`;
        if (!fresh && cache.has(key)) {
          res.statusCode = 200;
          res.setHeader("Content-Type", "application/json");
          res.setHeader("X-Sim-Cache", "hit");
          return res.end(cache.get(key));
        }

        const py = process.env.DAHS_PYTHON || "python";
        const t0 = Date.now();
        execFile(
          py,
          ["demo/build_run_log.py", "--seed", String(seed),
           "--baseline", baseline, "--stdout"],
          { cwd: repoRoot, maxBuffer: 64 * 1024 * 1024, timeout: 240000 },
          (err, stdout, stderr) => {
            if (err) {
              const tail = String(stderr || err.message || "").trim().slice(-700);
              return fail(500, `Simulation failed.\n${tail}`);
            }
            cache.set(key, stdout);
            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.setHeader("X-Sim-Cache", "miss");
            res.setHeader("X-Sim-Wall-Ms", String(Date.now() - t0));
            res.end(stdout);
          }
        );
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), dahsRunnerPlugin()],
  resolve: {
    alias: {
      "react/jsx-dev-runtime": path.join(nm, "react/jsx-dev-runtime.js"),
      "react/jsx-runtime": path.join(nm, "react/jsx-runtime.js"),
      "react-dom/client": path.join(nm, "react-dom/client.js"),
      "react-dom": path.join(nm, "react-dom/index.js"),
      react: path.join(nm, "react/index.js"),
      recharts: path.join(nm, "recharts"),
      "lucide-react": path.join(nm, "lucide-react"),
    },
  },
  optimizeDeps: { include: ["react", "react-dom", "recharts", "lucide-react"] },
  server: {
    port: 5180,
    fs: { allow: [".."] },
  },
});
