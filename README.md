# magnetar-component-smollm

SmolLM model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The component produces prefill, cached-decode and segmented decoder graphs
for SmolLM2 and SmolLM3 through the Magnetar `model-component-graph` 1.9.0
interface. The graph producer is derived from
`Magnetar-component-Llama` at `06bfbfad6bce156a449b02f790c3b6bf95417cab`;
the upstream copyright and MIT license are included in `LICENSE`.

The component reads the checkpoint's activation from the host and rejects
missing or non-SiLU activations before building a graph. Magnetar's engine
integration test builds and invokes the artifact using the
published SmolLM2-135M-Instruct dimensions for both prefill and decode. A
separate production-loading test compares a tiny, random Llama checkpoint's
prefill and cached-decode logits with Transformers. These are not tests of
the published checkpoint's weights.

The Rust library also reads and validates SmolLM3's per-layer RoPE/NoPE
position plan from a Hugging Face configuration. Magnetar passes the plan to
the Component, caches raw keys for NoPE layers, and compares tiny mixed-layer
prefill and cached-decode logits against Transformers. Integration tests also
exercise the published SmolLM3-3B configuration and malformed inputs.

## Built on the shared crates

The graph IR (`Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node`), the `GraphSink`
the builder writes to and the adapter to the contract's `graph-builder` come from
[`magnetar-component-sdk`](https://github.com/astorise/magnetar-component-sdk),
and the contract itself (the WIT file and its generated bindings, version 1.9.0)
from [`magnetar-component-api`](https://github.com/astorise/magnetar-component-api)
with its `v1_9` feature. Both are git dependencies pinned by commit while they
are in development, and published once stable. `src/plan.rs` keeps only what is
specific to this Component: the configuration and the graph, including SmolLM3's
rotary and NoPE layers; `src/component.rs` reads the Runtime's `model-config` and
calls the builder.

`tests/golden/` holds the graphs the builder emits, recorded before it moved onto
either crate (from the unmodified graph code, run on the host against a recording
stand-in for the Runtime's imports); `tests/golden_graphs.rs` compares every call
against them, and `tests/refusals.rs` pins what is refused. Regenerate the goldens
deliberately with `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.

## Work required

Loading and validating real released checkpoints remain necessary before
claiming full production qualification. SmolLM3 sliding-window attention and
MLP bias are currently refused; YaRN scaling is not implemented here.

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/smollm`.

`.github/workflows/qualification.yml` asks a different question: does the
Magnetar Runtime accept the artifact this Component builds? It calls Magnetar's
reusable
[`component-qualification.yml`](https://github.com/astorise/Magnetar/blob/main/.github/workflows/component-qualification.yml),
which builds this Component for the Component Model inside a Magnetar checkout,
in place of the one Magnetar pins, validates it and runs the Runtime's tests that
consume it (`tools/qualify_component.sh list`, in Magnetar, says which). The
tests stay in Magnetar; this repository only names the Component (`smollm`).

To build and validate the actual Component artifact:

```sh
cargo build --locked --release --target wasm32-unknown-unknown
wasm-tools component new target/wasm32-unknown-unknown/release/magnetar_component_smollm.wasm -o target/smollm2.component.wasm
wasm-tools validate target/smollm2.component.wasm --features component-model
```

The parent Magnetar component-integration workflow passes this freshly built
artifact to its engine graph test through `MAGNETAR_SMOLLM2_TEST_COMPONENT`.
