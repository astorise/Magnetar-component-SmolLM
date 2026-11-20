# magnetar-component-smollm

Placeholder for SmolLM model families in [Magnetar](https://github.com/astorise/Magnetar).

## Status

This is a scaffold only. It has no model graph, loader, weights, inference implementation, Component Model exports or supported checkpoint. Creating this repository does not make the model runnable in Magnetar.

## Work required

Checkpoint-specific decoder, tokenizer, attention and positional-encoding semantics, validated against reference logits.

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/smollm`.
