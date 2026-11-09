# ReCoKT

**Cooperative knowledge tracing with school withdrawal.**

ReCoKT lets schools train knowledge-tracing models together while keeping track of which schools contributed to each model component. When a school withdraws, the service removes components that depend on that school and updates the prediction route for the remaining schools.

This repository contains the research implementation of ReCoKT, three compatible knowledge-tracing backbones, and the membership-attack scoring code used in our privacy evaluation.

## 📚 What's inside

- **ReCoKT:** component training, prediction, withdrawal routing, and managed serving.
- **Backbones:** DKT, SAKT, and KeenKT.
- **Privacy attacks:** school-level Gaussian scoring and DKT student-level LiRA/RULI scoring.
- **Tests:** synthetic training and prediction checks, plus withdrawal and attack-scoring tests.

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

The smoke test creates temporary synthetic data, trains a pairwise component for each backbone using the 6+6 schedule, and checks loading, prediction, and withdrawal routing. The selective-state tests use mock streams.

## 🗂️ Prepare your data

Download the datasets from ASSISTments and prepare them under their original terms:

- [ASSIST2009 Non-Skill-Builder data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/non-skill-builder-data-2009-10)
- [ASSIST2012–2013 school data](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)

To train a component bank, prepare:

1. **`manifest.json`** — skill count, ordered schools, each school's training-file path, and training counts (`n_windows` and `emitted_targets`).
2. **`sha256.json`** — a `files` index mapping each relative input path to its SHA-256 hash.
3. **`component_plan.json`** — the frozen data hashes and component jobs, including the ordered school ring, component specifications, and seeds.

Each training JSONL record must follow `recokt_vendor.data._validate_record`: include a school ID, pseudonymous student ID, aligned `skills` and `responses`, causal within-window `history`, and `sample_ids`. The original training recipe expects school IDs as numeric strings. Prepare a 19-school target registry for withdrawal evaluation.

## 🏋️ Train and evaluate

Train a bank for one dataset, backbone, and seed:

```bash
recokt-bank train --dataset as09 --arch dkt --seed 1 \
  --manifest /path/to/data/manifest.json \
  --index /path/to/data/sha256.json \
  --plan /path/to/component_plan.json \
  --output /path/to/new-bank
```

The command checks the input hashes and 20-school ring, then writes component directories and `bank.json`. It resumes completed components whose receipts and hashes match, and keeps partial output for inspection if a component fails. The bank includes both the original trained factors and the quantized component used by the loader.

To evaluate a withdrawal using a separately prepared, frozen 19-school target registry:

```bash
recokt-bank evaluate --bank /path/to/new-bank/bank.json \
  --registry /path/to/registry.json --withdrawn 2770 \
  --output /path/to/new-case-summary.json
```

This checks target identity and coverage, and reports per-school AUC and student-balanced Brier.

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

## 🧪 Reproduction status

The paper's main protocol uses two datasets, 20 schools, three backbones, five seeds, and five fixed single-school withdrawals per dataset. See [source provenance](PROVENANCE.md) and the [release checklist](RELEASE_CHECKLIST.md) for implementation details and verification status.

## 📄 License

The project's original code is provided under the [MIT License](LICENSE).
