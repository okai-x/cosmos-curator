# video2scene-build

Build Harbor tasks from immutable `video2sceneRL-ver` bundles. Ray Data parallelizes
bundles; the existing factory owns each bundle's ordered derivation, calibration,
fixture acceptance and packaging. The factory runs as a local Python subprocess,
with local Blender subprocesses and no temporary Docker containers. The subprocess
boundary keeps its module-global state and workspace private to each bundle. Canonical outcomes are published to Lance through
shared append. No new stage hierarchy or distributed inner-build DAG is introduced.

## Run

```bash
cosmos-curator pipeline template video2scene-build > video2scene-build.yaml
python -m cosmos_curator.next.recipes.video2scene_build.snapshot /opt/video2sceneRL-ver
python -m cosmos_curator.next.recipes.video2scene_build.snapshot --runtime
```

Put these values in `build.source_digest` and `build.python_runtime_digest`. Run the
runtime fingerprint command using the same Python and `PATH` as the worker. The
digest includes Python, numpy, OpenCV, Pillow, Jinja2 and psutil versions, plus the
complete `ffmpeg -version` output. FFmpeg must be available on the worker PATH. Fill in the bundle paths, source checkout, Blender executable,
exact Blender release and output root. The template's all-zero digests are placeholders.

```bash
cosmos-curator pipeline validate video2scene-build.yaml
pixi run --as-is run-pipeline video2scene-build.yaml
```

The source checkout must support `factory/build.py --backend local` and forward
`Build.judge()`'s `--parallel` to `accept_judge.py`. Both acceptance paths run with
parallelism one. No Docker executable, daemon or scorer image is required for
construction. Old configurations containing `build.scorer_image` are rejected;
remove that field, recompute digests and use a new output root.

`build.judge` defaults to false. Setting it to true enables the factory's paid VLM
calibration and acceptance, with the key supplied on each worker through `V2S_VLM_KEY`
or `V2S_VLM_KEY_FILE`. `judge_max_usd` is the upstream acceptance cap, **not** a total
build budget: calibration is separate, and interrupted attempts can repeat charges.
No credentials are stored in the configuration or result contract.

## Runtime and storage contract

- Inputs, source, output and executables are local/shared filesystem locations.
  All workers and replacement drivers must see them at the same absolute paths.
  Direct S3/GCS payload staging is not implemented. The shared filesystem must
  support atomic directory rename within `output.root/payloads`.
- Each managed worker node advertises `video2scene_build_slot: 1`. Local Ray startup
  advertises one slot automatically. Each build reserves that slot, `build.cpus`
  CPUs and `execution.memory_gib` logical memory. Parallelism is across nodes.
  The slot serializes this recipe's builds; it cannot prevent unrelated workloads.
- Use dedicated, homogeneous worker nodes and identify their controlled measurement
  conditions in `build.measurement_class`. AC-14 latency affects task release, so
  CPUs and measurement class belong to the result contract. The local factory
  limits its inherited CPU affinity to `build.cpus`. Logical Ray resources
  do not enforce physical exclusivity or child-process memory limits.
- Native AC-14 memory measurement is `sampled_process_tree_rss`: the sum of live
  verifier/descendant RSS sampled every 0.05 seconds. It can miss shorter peaks and
  counts shared pages in multiple processes; it is not the former cgroup peak.
  Acceptance thresholds remain unchanged, but timing and memory results must be
  compared within the same execution contract. There is no native hard RAM quota.
- The source and reference scripts are trusted, as in the original factory. Host
  Python/Blender run reference and generated fixture code without a sandbox.
  Construction and factory acceptance are for trusted bundles only. Untrusted
  Agent submissions are scored later in Harbor with its required isolation. Both
  `BLENDER` and `V2S_BLENDER` are set to the selected executable; ambient `V2S_*` overrides are removed except judge keys.
- Use the factory's matching agent/verifier base images when consuming packaged
  tasks with Harbor. Native construction does not build, publish or replace the
  base image references embedded by the factory's policy; packaged Agent execution
  and verifier isolation stay owned by Harbor.

## Identity and publication

Discovery hashes every bundle file, refuses links/special files, and freezes the
bundle location, content digest and sample name. `build_id` combines sample, content
digest and the result contract. Duplicate inputs with the same identity are deduplicated.
Source identity covers `factory/`, `engine/` and `tools/`, including policies and
templates. Python caches are excluded. Both source and input are verified again after
copying to the worker's independent workspace.

One result row represents one build. `tasks` holds the profile, factory state,
packaged task path, complete scorecard JSON and state explanation. Intake returns
have an empty task list and remain queryable. Held/quarantined tasks are retained;
consumers select task entries with `state == "released"` for delivery. A requested
full profile that the factory cannot package is recorded as `not_generated`.

Every attempt starts under `payloads/.attempts/<build_id>-<unique>/`. It contains a
source snapshot, copied bundle, factory output, logs and a completion manifest.
After validation, atomic directory rename publishes it at `payloads/<build_id>/`.
Only then does the Ray task return a result for checkpointing. Replay validates and
reuses an existing completed payload; it never overwrites it. Failed or losing
attempts remain for diagnosis and may be removed by an operator after the run.

Shared append validates staged outcomes and publishes each selected set in one
Lance transaction to `builds.lance`. No worker runs the factory's destructive
`--collect` command. The output table and its recovery workspace require **one active
driver**; the shared helpers provide no writer lease. Select a bounded group with
`execution.max_items` when earlier result visibility is needed.

An output table is bound to one result contract. Changing source, Python runtime,
Blender, FFmpeg, execution backend, judge settings or measurement conditions requires
a new output root. Checkpoint packing and fragment sizing can change across recovery. Keep inputs,
published payloads and Lance physical layout immutable; do not compact/delete rows
under the current shared recovery contract.

## Failures and recovery

Explicit failed intake is a terminal `returned` result. Factory quality decisions
remain the factory's decisions. Nonzero build exits without a validated intake
return, incomplete tasks, scorer errors, missing rewards, mismatched tools and
storage failures propagate without a terminal row. Rerun the same configuration
after fixing an infrastructure problem. Worker-loss retries use fresh attempts;
application errors are not promised unlimited automatic retries.

AC-14's `full_without_judge` run deliberately stops at the missing-key error after
rendering. This specific expected stop is distinguished from actual timing/scorer
failures, including subprocess failures that produce no `error.json`.
A reference refused by the verifier's gate can instead finish normally with zero
reward before reaching the judge; that remains a business outcome.

Recovery is per completed bundle, not per internal factory step. Incomplete builds
restart from the beginning, avoiding the upstream error-cache behavior. There is
no adapter-wide wall-clock deadline; upstream verifier limits still apply. Managed
deployments must clean up subprocess trees after worker loss or cancellation.

Tests use the shared append conformance suite and a real Ray fresh-cluster scenario
with real Python subprocesses producing the factory's file protocol. They cover
publication, payload replay, input mutation, terminal/retryable outcomes and the
intentional timing error. A native subprocess test disables Docker with a failing
executable and exercises real runtime preflight. These tests do not establish full
Blender/scorer correctness, multi-node performance, or managed-worker join/loss/requeue qualification.
