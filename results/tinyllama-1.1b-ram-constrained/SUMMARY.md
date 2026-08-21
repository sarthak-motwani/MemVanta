# TinyLlama 1.1B RAM-constrained experiment

> **Invalid / superseded run. Do not use these numbers as benchmark evidence.**
>
> Run `32451307562` used `systemd-run` without entering the repository working directory. The benchmark executables therefore did not actually execute under the requested memory ceilings; the tiny recorded `MemoryPeak` values confirm the harness failure.

A corrected cgroup-v2 `MemoryMax` workflow has been committed and is rerunning the experiment with an explicit working directory, captured service status, a valid high-memory baseline requirement, and an expanded 1536→512 MiB sweep.

The next successful publication will replace this directory with valid evidence.
