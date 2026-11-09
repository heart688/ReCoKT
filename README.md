# ReCoKT

**Cooperative knowledge tracing with school withdrawal.**

ReCoKT lets schools train knowledge-tracing models together while keeping track of which schools contributed to each model component. When a school withdraws, the service removes components that depend on that school and updates the prediction route for the remaining schools.

This repository contains the research implementation of ReCoKT, three compatible knowledge-tracing backbones, and the membership-attack scoring code used in our privacy evaluation.

## 📚 What's inside

- **ReCoKT:** component training, prediction, withdrawal routing, and managed serving.
- **Backbones:** DKT, SAKT, and KeenKT.
- **Privacy attacks:** school-level Gaussian scoring and DKT student-level LiRA/RULI scoring.
- **Tests:** synthetic training and prediction checks, plus withdrawal and attack-scoring tests.

The code runs federated training as a **single-machine, multi-process simulation**. It does not implement communication over a real network.

## 🧩 How ReCoKT works

Each school trains a local component and cooperates with its two neighbors to train two pairwise components.

Each component uses a rank-16 backbone adaptation and a correction head. Training uses six backbone rounds followed by six head rounds. Unless a sequential-withdrawal experiment is explicitly set up, each request starts from the same source bank.

## 🚀 Getting started

You need **Python 3.10 or later**, PyTorch 2.4 or later, and NumPy. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the checks:

```bash
python tests/smoke.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

The smoke test creates temporary synthetic data and trains a pairwise component for each backbone using the full 6+6 schedule. It checks loading, causal prediction and withdrawal routing, and also exercises DKT selective-state withdrawal. These synthetic outputs are **not** paper results. The separate selective-state tests use mock streams.

## 🗂️ Prepare your data

This repository does **not** include datasets, student records, prediction files, trained weights, or the paper's split and target registries. Obtain and prepare the inputs separately under their original terms. Do not use test data to select training settings.

Download the datasets from ASSISTments:

- [ASSIST2009 Non-Skill-Builder data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/non-skill-builder-data-2009-10)
- [ASSIST2012–2013 school data](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)

To train a component bank, prepare:

1. **`manifest.json`** — skill count, ordered schools, each school's training-file path, and training counts (`n_windows` and `emitted_targets`).
2. **`sha256.json`** — a `files` index mapping each relative input path to its SHA-256 hash.
3. **`component_plan.json`** — the frozen data hashes and component jobs, including the ordered school ring, component specifications, and seeds.

Each training JSONL record must follow `recokt_vendor.data._validate_record`: include a school ID, pseudonymous student ID, aligned `skills` and `responses`, causal within-window `history`, and `sample_ids`. The original training recipe expects school IDs as numeric strings. The prepared data, plan, and 19-school evaluation registry must be supplied by the user; ready-made copies are not bundled.

## 🏋️ Train and evaluate

Train a bank for one dataset, backbone, and seed:

```bash
recokt-bank train --dataset as09 --arch dkt --seed 1 \
  --manifest /path/to/data/manifest.json \
  --index /path/to/data/sha256.json \
  --plan /path/to/component_plan.json \
  --output /path/to/new-bank
```

The command checks the input hashes and 20-school ring, then writes component directories and `bank.json`. It resumes only completed components whose receipts and hashes match. If a component fails, its partial output is kept for inspection; remove or move **only your own failed output** before retrying. Both the original trained factors and the quantized component used by the loader are saved.

To evaluate a withdrawal using a separately prepared, frozen 19-school target registry:

```bash
recokt-bank evaluate --bank /path/to/new-bank/bank.json \
  --registry /path/to/registry.json --withdrawn 2770 \
  --output /path/to/new-case-summary.json
```

This checks target identity and coverage, and reports per-school AUC and student-balanced Brier. It does not select hyperparameters, aggregate across seeds, run baselines, or generate paper tables. Its real-data predictions have not yet been independently matched against the original experiment.

For lower-level use, run `recokt-component --help` or use `recokt.predict` to load components, build routes, and make predictions.

## 🔐 Run a privacy audit

The attack utilities are in [`recokt.attacks`](src/recokt/attacks/README.md):

- **School-model audit:** score Pre or Post against a matched Clean model using five-seed paired Gaussian scoring.
- **Student-membership audit:** score the paper's DKT LiRA and RULI attacks using independently prepared reference, calibration, and evaluation systems.

For example, once the private student-audit inputs have been prepared:

```bash
recokt-student-attack-score --dataset as09 --input-dir /private/audit \
  --output /private/audit-summary.json
```

The [attack guide](src/recokt/attacks/README.md) explains the expected files, query features, and scoring rules. Shadow-system training, private trajectories, and the original server's full input audit are **not** bundled. Results apply to the evaluated attack and query interface; they are not a general privacy guarantee.

## 🔄 Serve predictions after withdrawal

First create a serving copy of a complete component bank:

```bash
recokt-export-serving --bank /path/to/new-bank/bank.json \
  --output /path/to/new-serving-copy
```

For DKT, open a student session, process a partner school's withdrawal, and continue prediction:

```python
from recokt.factor_stream import FactorStream
from recokt.serving.managed_quant_serving_factory_v1 import make_session_factory
from recokt.serving.selective_state_policy_v2 import (
    SelectiveManagedServingBundle, exact_stream_type_guard,
    make_quantized_dkt_factory,
)

factory = make_quantized_dkt_factory(state_reuse_guard=exact_stream_type_guard(FactorStream))
with SelectiveManagedServingBundle('/path/to/new-serving-copy', factory) as service:
    service.open_session('session-1', 'retained-school-id', make_session_factory('dkt'))
    service.observe_session('session-1', 0, 1)  # skill index, observed response
    result = service.withdraw('partner-school-id')
    if 'session-1' in result['sessions_replay_required']:
        raise RuntimeError('Reopen and replay the retained student history')
    probability = service.predict_session('session-1', 1)
```

The managed bundle changes its **own serving copy**, not the original training bank or unrelated files. `FactorStream` is the reviewed state-isolated DKT implementation; other stream types fall back to replay unless separately verified.

## 🧪 Reproduction status

The paper's main protocol uses two datasets, 20 schools, three backbones, five seeds, and five fixed single-school withdrawals per dataset. The original prepared data, experiment plans, and registered targets are needed to reproduce it and are not included here.

The portable wrappers were assembled from the research implementation. Backbone and training-source hashes have been checked against the recorded server manifest, and the attack scorer has been compared with the original scorer on synthetic inputs. Full real-data training, predictions, attack inputs, and metric aggregation still need an end-to-end comparison. See [source provenance](PROVENANCE.md) and the [release checklist](RELEASE_CHECKLIST.md).

## 📄 License

The project's original code is provided under the [MIT License](LICENSE). The KeenKT implementation is adapted from [HubuKG/KeenKT](https://github.com/HubuKG/KeenKT), which is distributed under Apache-2.0; its license is included in [KEENKT_APACHE-2.0.txt](KEENKT_APACHE-2.0.txt). The ReCoKT wrapper modifies the original behavior.
