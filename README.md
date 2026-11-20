# magnetar-component-smollm

SmolLM model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The initial Rust library reads and validates SmolLM3's per-layer RoPE/NoPE
position plan from a Hugging Face configuration. Integration tests exercise the
published SmolLM3-3B configuration at a pinned revision and malformed inputs.
The library does not load weights or implement graph production, Component
Model exports, or inference; SmolLM3 is not yet runnable in Magnetar.

## Work required

Checkpoint-specific decoder, tokenizer, attention and positional-encoding
semantics still need integration into Magnetar and validation against reference
logits. The runtime contract needed for per-layer RoPE/NoPE is tracked in
[Magnetar #212](https://github.com/astorise/Magnetar/issues/212).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/smollm`.
