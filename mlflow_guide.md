# MLflow Guide

A practical reference for using MLflow in this project.

---

## 1. Mental model

MLflow has three pieces you need to know:

- **Tracking server** — a long-running web server that stores runs and serves
  the UI. Started once, used by many training scripts.
- **Experiment** — a named bucket of related runs (e.g. `GPT (Decoder Only)`).
- **Run** — one execution of your training script. Holds params, metrics,
  artifacts, system metrics.

Your training script *talks to* the tracking server over HTTP at
`http://localhost:5000`. If the server is down, the script hangs/retries.

---

## 2. This project's setup

- Tracking URI: `http://localhost:5000`
- Backend store: `sqlite:///mlflow.db` (in project root — keeps runs/params/metrics)
- Artifact store: `mlartifacts/` (created on first artifact log)

---

## 3. Daily workflow

```text
[once per boot] start server
       ↓
[every run]   python train.py     # logs to server
       ↓
[anytime]     open localhost:5000 # browse / compare runs
```

You do **not** restart the server between training runs. It stays up.

---

## 4. Server lifecycle

### Start (recommended — survives closing the terminal)

```bash
nohup uv run mlflow server --host 127.0.0.1 --port 5000 > mlflow.log 2>&1 &
```

Runs in the background, logs to `mlflow.log`, keeps running until the machine
reboots or you kill it. Open the UI at <http://localhost:5000>.

### Start (foreground — dies when you close the terminal)

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Useful when you want to watch the server logs live.

### Check if it's running

```bash
pgrep -af "mlflow server"
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/   # expect 200
```

### Stop

```bash
pkill -f "mlflow server"
```

---

## 5. Logging API cheatsheet

Inside `with mlflow.start_run(...):`

| Call | Use it for |
| --- | --- |
| `mlflow.log_param(key, value)` | Hyperparameters fixed for the run (lr, batch_size). One value per key. |
| `mlflow.log_params({...})` | Many params at once. |
| `mlflow.log_metric(key, value, step=i)` | Time-series numbers (loss, accuracy). Plotted in UI. |
| `mlflow.log_metrics({...}, step=i)` | Many metrics at the same step. |
| `mlflow.log_artifact(path)` | A file (plot, checkpoint, config) attached to the run. |
| `mlflow.log_artifacts(dir)` | A whole directory. |
| `mlflow.set_tag(key, value)` | Free-form labels (`stage=baseline`). |
| `mlflow.enable_system_metrics_logging()` | Auto-log CPU/GPU/RAM during the run. |

Rule of thumb: **params = inputs, metrics = outputs, artifacts = files.**

---

## 6. Minimal training script with MLflow

A small, runnable example you can copy as a starting point.

```python
import mlflow
import torch
from torch import nn, optim

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("toy-linear-regression")
# Tag as traditional ML so the UI shows the normal Runs view, not GenAI tabs.
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")

# Toy data: y = 3x + 2 + noise
x = torch.linspace(-1, 1, 200).unsqueeze(1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

config = {"lr": 1e-2, "epochs": 50, "hidden": 16}

with mlflow.start_run(run_name="baseline"):
    mlflow.log_params(config)

    model = nn.Sequential(
        nn.Linear(1, config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], 1),
    )
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # step-level metric — shows up as a line plot in the UI
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    # final summary metric + checkpoint as an artifact
    mlflow.log_metric("final_loss", loss.item())
    torch.save(model.state_dict(), "model.pt")
    mlflow.log_artifact("model.pt")
```

After running, open <http://localhost:5000>, click the experiment, then the
run — you'll see params, the `train_loss` curve, and `model.pt` under
Artifacts.

---

## 7. Viewing & comparing in the UI

- **Experiment list** (left panel): one row per experiment.
- **Run table**: rows are runs, columns are params/metrics. Tick rows.
- **Compare** button: side-by-side params, overlaid metric plots.
- **Run detail**: click a run → tabs for parameters, metrics, artifacts, system
  metrics.

---

## 8. Maintenance

### Set the experiment "kind" (avoid the GenAI view)

MLflow 3 has two experiment kinds, selected by the tag
`mlflow.experimentKind`:

- `custom_model_development` — traditional ML training → normal **Runs** view.
- `genai_development` — LLM/agent workflows → **Traces / Evaluation runs /
  Labeling sessions** view.

If the tag isn't set, MLflow *infers* the kind from your code/run names, and
sometimes guesses GenAI for training scripts — your runs then show up under
"Evaluation runs" instead of "Runs". Set it explicitly to avoid this.

In code (do this once per experiment, right after `set_experiment`):

```python
mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
```

Or, retroactively on an existing experiment (server must be up):

```bash
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/set-experiment-tag \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"1","key":"mlflow.experimentKind","value":"custom_model_development"}'
```

Refresh the UI to pick up the change.

### Garbage-collect soft-deleted entries

Keeps active experiments, permanently drops anything trashed via the UI:

```bash
.venv/bin/mlflow gc --backend-store-uri sqlite:///mlflow.db
```

### Delete one experiment (server must be up)

```bash
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/delete \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"1"}'
```

### Restore a soft-deleted experiment

The UI's "Delete" only soft-deletes — the experiment still appears in
`get_experiment_by_name(...)` with `lifecycle_stage='deleted'`, which makes
`mlflow.set_experiment(name)` fail. Restore it:

```bash
curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/restore \
  -H "Content-Type: application/json" \
  -d '{"experiment_id":"1"}'
```

List experiments (including deleted ones) to find the id:

```bash
curl -sS http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results":50,"view_type":"ALL"}'
```

### Full reset (wipes everything)

```bash
pkill -f "mlflow server"
rm -f mlflow.db
rm -rf mlartifacts mlruns
nohup uv run mlflow server --host 127.0.0.1 --port 5000 > mlflow.log 2>&1 &
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Script hangs ~30s then `ConnectionRefusedError: [Errno 111]` | Server not running | Start it (§4). |
| Training runs appear under "Evaluation runs" / "Traces" in the UI | Experiment kind inferred as GenAI (tag not set) | Set `mlflow.experimentKind=custom_model_development` (§8). |
| `Cannot set a deleted experiment ... as the active experiment` | Experiment was soft-deleted in UI | Restore it (§8) or use a different name. |
| Script can't find `mlflow.db` / fresh state on every run | Server started from a different cwd | Always start from project root, or use absolute `--backend-store-uri`. |
| UI loads but runs are missing | Connected to wrong tracking URI in code | Check `mlflow.set_tracking_uri(...)` matches the server. |

### Quick smoke test

```bash
.venv/bin/python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('smoke')
with mlflow.start_run(run_name='smoke'):
    mlflow.log_param('ok', 1)
    mlflow.log_metric('val', 0.42)
"
```

If this prints a run URL without hanging, the server is healthy.
