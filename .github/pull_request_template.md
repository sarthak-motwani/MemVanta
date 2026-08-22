## Summary

Describe the change and why it is needed.

## Change type

- [ ] Runtime/code change
- [ ] Benchmark/reproducibility change
- [ ] Documentation only
- [ ] CI/build change

## Validation

- [ ] Release build succeeds
- [ ] `ctest` passes
- [ ] Existing deterministic correctness checks pass
- [ ] No benchmark number was changed without raw evidence

## Performance / memory evidence

If this PR changes runtime behavior, include before/after measurements using the same model, workload, threads, context, batch and KV-cache settings.

- Peak RSS before:
- Peak RSS after:
- Prompt-processing throughput before/after:
- Token-generation throughput before/after:
- Model SHA-256:
- MemVanta base/head commit SHAs:
- Comparison-runtime commit SHA, if applicable:

Attach or link raw outputs. Follow `docs/MEMORY_BENCHMARKING.md` and `docs/BENCHMARK_CHECKLIST.md` for publishable claims.

## Claim discipline

- [ ] Any memory claim is scoped to the tested model/workload/hardware
- [ ] Throughput is reported transparently alongside memory results
- [ ] Lowest successful cgroup ceiling is described as a tested boundary, not an exact RAM requirement
- [ ] Contradictory or failed runs are retained and explained

## Reviewer notes

Call out anything that deserves extra scrutiny, especially benchmark methodology, correctness, numerical tolerance, or changes to `results/` and benchmark workflows.
