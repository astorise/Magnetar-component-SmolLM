# magnetar-component-smollm

SmolLM model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The component emits prefill, cached-decode and segmented decoder graphs for
the Llama-compatible SmolLM2 architecture through the Magnetar
`model-component-graph` 1.4.0 interface. The graph producer is derived from
`Magnetar-component-Llama` at `06bfbfad6bce156a449b02f790c3b6bf95417cab`;
the upstream copyright and MIT license are included in `LICENSE`.

Magnetar's engine integration test builds and invokes the component using the
published SmolLM2-135M-Instruct dimensions for both prefill and decode. A
separate production-loading test compares a tiny, random Llama checkpoint's
prefill and cached-decode logits with Transformers. These are not tests of
the published checkpoint's weights. The checkpoint's activation must also be
preserved and validated by the runtime; see
[Magnetar #217](https://github.com/astorise/Magnetar/issues/217).

The Rust library also reads and validates SmolLM3's per-layer RoPE/NoPE
position plan from a Hugging Face configuration. Integration tests exercise the
published SmolLM3-3B configuration at a pinned revision and malformed inputs.
SmolLM3 graph production and inference are not implemented.

## Work required

Loading and validating real released checkpoints remain necessary before
claiming production inference support. The runtime contract needed for
SmolLM3 per-layer RoPE/NoPE is tracked in
[Magnetar #212](https://github.com/astorise/Magnetar/issues/212).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/smollm`.

To build and validate the actual Component artifact:

```sh
cargo build --locked --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_component_smollm.wasm -o target/smollm2.component.wasm
wasm-tools validate target/smollm2.component.wasm --features component-model
```

The parent Magnetar component-integration workflow passes this freshly built
artifact to its engine graph test through `MAGNETAR_SMOLLM2_TEST_COMPONENT`.
