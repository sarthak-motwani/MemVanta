# Benchmark Publication Checklist

Before treating a result as publishable:

- [ ] Same GGUF file used by both runtimes
- [ ] Model SHA-256 recorded
- [ ] MemVanta commit SHA recorded
- [ ] Comparison runtime commit SHA recorded
- [ ] CPU-only settings matched
- [ ] Thread count matched
- [ ] Context size matched
- [ ] Batch size matched
- [ ] Prompt/generation token counts matched
- [ ] KV-cache precision matched where possible
- [ ] At least one warm-up run separated from measurements
- [ ] At least five measured runs retained
- [ ] Raw outputs committed
- [ ] Mean, standard deviation, median, minimum, and maximum reported for peak RSS
- [ ] Throughput shown beside memory results
- [ ] Swap state recorded
- [ ] cgroup limits and OOM outcomes retained for constrained-memory tests
- [ ] Lowest successful tested ceiling described as a tested boundary, not an exact RAM requirement
- [ ] Hardware/OS/compiler metadata captured
- [ ] Result wording scoped to the tested model/workload/hardware
- [ ] Any failed or contradictory runs retained and explained
