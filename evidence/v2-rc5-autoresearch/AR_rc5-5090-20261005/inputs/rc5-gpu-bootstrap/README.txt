PDH-Zero rc5 GPU bootstrap bundle

Purpose:
- Replace only the GPU-startup-critical cache/candidate from the truncated full evidence attachment.
- This is NOT the complete rc5 evidence archive and is NOT a promoted model.

Files:
- auxiliary-v2.pt
  SHA256 8754e41f38f43824aa98a4d397aef0a04c012f07bbd91819bc7787b94dd3fee0
  Frozen TRAIN-side auxiliary dataset: 512 complete natural hands / 13,953 sampled points.
- probe-candidate-seed-132846611674848.pt
  SHA256 a4ff8094892bb257e1a72173b0040a7a5d21069067a8921f07428288740b789e
  Existing CPU-smoke joint candidate; use only for CUDA integration/parallel consistency probe.
  It was NOT promoted and must NOT replace d000000266137.
- h3-smoke-report.json / probe-candidate-trial.json
  Provenance for the smoke candidate.

The target instance must continue to use its preserved rc4.1 TRAIN/HELD archives and the frozen baseline d000000266137.
