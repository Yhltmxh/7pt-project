# Derm7pt development

Read `derm7pt_ai_handoff.md`, `README.md` and `docs/protocol.md` before changes.
Work only in this project; the ISIC2025 project is a read-only reference.

- Five diagnosis labels have fixed order BCC, NEV, MEL, MISC, SK.
- Data returns inputs, targets and ids separately. Only inputs enter forward.
- Default baselines use images, CE and final EMA; no metadata or attributes.
- Preserve official row-index splits. Never use test for development or training.
- Draft settings and synthetic smoke results are not formal experimental evidence.
- Keep one training and inference entry point. Add fusion through the registry.
- Refuse unknown labels/config keys, incompatible checkpoints and partial reuse.
- Record source/config/data/checkpoint hashes and actual validation limits.
- Do not start a full training suite or change another repository without instruction.

Run `python -m pytest`, `python scripts/smoke_test.py`, and
`python scripts/run_experiments.py --suite configs/experiments/baselines.yaml --dry-run`.
