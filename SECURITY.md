# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.

Use GitHub's **Private vulnerability reporting / Security advisory** flow for this repository so details can be reviewed before public disclosure.

When reporting, include:

- affected version or commit
- minimal reproduction steps
- impact assessment
- platform/compiler information
- suggested mitigation, if known

We will triage reports as quickly as practical and coordinate disclosure after a fix or mitigation is available.

## Scope

Security-sensitive areas include GGUF parsing, tokenizer handling, model-file bounds checking, memory mapping, quantized kernel indexing, thread synchronization, and any future network-facing components.
