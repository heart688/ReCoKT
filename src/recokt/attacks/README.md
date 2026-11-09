# Offline membership audits

These utilities implement the paper's school-model Gaussian scoring and the
expanded DKT student LiRA/RULI scoring. They operate on predictions or
precomputed features. They do **not** supply the dataset, trained shadow
systems, target registries, frozen student cohorts, or recorded predictions.
Attack results describe the chosen interface and cohort; they are not a
general privacy guarantee.

`features.trajectory_features` converts one student's aligned, causal target
probabilities and binary responses into the two features used in the DKT
study. For ReCoKT, first fuse component probabilities for the chosen retained
school with `recokt.predict.predict_route`; do not average logits. Keep query
trajectories fixed across Original, Post and independently trained Clean.
`features.query_route` performs this route prediction and feature extraction
for a caller-supplied cohort without writing student-level outputs.
`design.balanced_masks` recreates the frozen paired IN/OUT assignment given
the ordered, disjoint 64 calibration and 64 evaluation student IDs. Selecting
eligible students and training systems is the caller's responsibility; every
window for a given student must be included or excluded together.

`school.paired_five_seed` consumes five matched school-level aggregate features
for Pre or Post and five corresponding independently trained Clean features.
It fits one Gaussian per class using the other four seed pairs, shrinks each
sample variance halfway to their pooled variance and compares each held-out
pair using fixed log-density ratio direction. `school_feature` averages
student mean log likelihoods with equal weight. To match the reported table,
average the five withdrawal results within each seed, then report the mean
and sample standard deviation across seeds. Do not pool the held-out pairs
into an unpaired 5-by-5 AUC.

The portable student scorer preserves the fixed 88-system scoring rule from
`attack_expanded_score_v3.py`: 64 reference, eight calibration, and 16
evaluation systems; 64 distinct calibration and 64 evaluation students; LiRA
Gaussian scores, RULI kernel-density scores, threshold at 10% calibration
FPR, and a 500-replicate student-pair/system cluster bootstrap. The RULI
efficacy challenge uses one Post or Clean output per pair. This is the DKT
one-retained-route study, not a general attack evaluation of other backbones.

```bash
recokt-student-attack-score --dataset as09 --input-dir /private/audit \
  --output /private/audit-summary.json
```

The input directory uses the original offline study layout:

* `plan_private.json`: `dataset`, `candidates`, `calibration`, `evaluation`,
  and 88 `jobs` with one `masks` object mapping student IDs to 0/1 each.
* `systems/shadow00` through `shadow87`: `query_private.json` mapping each of
  `Original`, `Post`, and `Clean` to every candidate's `mean_loglik` and
  `mean_true_label_logodds`; `query_commit.json` contains `status` =
  `complete`, `plan_sha256` and `queries_sha256` of those exact files.

The scorer checks those file hashes but does **not** itself prove that the
models, train/holdout membership, query routes, or outputs are authentic.
The original server's separate input audit, source-model checks, private
cohort and system training pipeline are not packaged as a reproducible
end-to-end experiment. Validate those inputs and their use permissions
independently. The input and its hashes can be sensitive; do not publish them.

The scorer is a path-portable adaptation of the original scoring code, with
the server's process-control and scorer-file hash checks removed. Synthetic
tests alone do not establish real-data end-to-end numerical equivalence.
