//! Golden graphs: every call the builder makes, recorded as text and compared
//! with the files in `tests/golden/`.
//!
//! These files were recorded from the builder before it moved onto
//! `magnetar-component-sdk` and `magnetar-component-api`: the unmodified graph
//! code of `src/llama_graph.rs` at `af99a5c`, run on the host against a
//! recording stand-in for the `graph-builder` and `model-config` imports its
//! generated bindings provide. They pin the graph a Model Instance receives: a
//! change that alters one node, attribute, input order, shape or KV involvement
//! fails here, whoever made it. Regenerate them deliberately with
//! `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.
//!
//! The test uses only the crate's public `plan` API, so it holds across the
//! change of the types behind it.

use magnetar_component_smollm::{
    PositionPlan,
    plan::{
        BandedRopeScaling, Config, GraphSink, KvRole, Node, Phase, RopeScaling, Segment, Value,
        build_graph,
    },
};
use std::{fmt::Write as _, path::PathBuf};

/// Records every call, one line each, in call order.
#[derive(Default)]
struct TextSink {
    text: String,
}

impl TextSink {
    fn line(&mut self, line: std::fmt::Arguments<'_>) {
        writeln!(self.text, "{line}").unwrap();
    }
}

impl GraphSink for TextSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        self.line(format_args!(
            "begin {phase:?} sequence={sequence_length} offset={position_offset}"
        ));
    }
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.line(format_args!("input {name} {shape:?}"));
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        self.line(format_args!("weight {name}"));
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.line(format_args!("alias {edge} -> {target}"));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        self.line(format_args!("kv {layer} {role:?}"));
        format!("kv:{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let mut line = format!(
            "node {} op={} family={} inputs={:?} shape={:?}",
            node.id, node.operator, node.family, node.inputs, node.shape
        );
        for attr in &node.attrs {
            let value = match &attr.value {
                Value::Bool(v) => format!("bool:{v}"),
                Value::Int(v) => format!("int:{v}"),
                Value::Float(v) => format!("float:{v:?}"),
                Value::Text(v) => format!("text:{v}"),
            };
            write!(line, " {}={value}", attr.name).unwrap();
        }
        if let Some(kv) = &node.kv {
            write!(
                line,
                " kv=({}, {:?}, append={})",
                kv.resource, kv.role, kv.append
            )
            .unwrap();
        }
        self.line(format_args!("{line}"));
        format!("out:{}", node.id)
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.line(format_args!("finish {output}"));
        output.into()
    }
}

fn tiny(
    layers: u32,
    attention_bias: bool,
    tie_word_embeddings: bool,
    rope_layers: Option<Vec<bool>>,
) -> Config {
    Config {
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: layers,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dimension: 4,
        vocabulary_size: 17,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings,
        attention_bias,
        hidden_activation: Some("silu".into()),
        rope_scaling: None,
        rope_layers,
    }
}

/// A published checkpoint's `config.json`, as the Runtime would bind it:
/// `rope_layers` is the position plan of a SmolLM3 configuration.
fn published(raw: &[u8], head_dimension: Option<u64>, rope_layers: Option<Vec<bool>>) -> Config {
    let raw: serde_json::Value = serde_json::from_slice(raw).unwrap();
    let heads = raw["num_attention_heads"].as_u64().unwrap();
    let hidden = raw["hidden_size"].as_u64().unwrap();
    assert!(raw["rope_scaling"].is_null());
    Config {
        hidden_size: hidden,
        intermediate_size: raw["intermediate_size"].as_u64().unwrap(),
        num_hidden_layers: raw["num_hidden_layers"].as_u64().unwrap() as u32,
        num_attention_heads: heads as u32,
        num_key_value_heads: raw["num_key_value_heads"].as_u64().unwrap() as u32,
        head_dimension: head_dimension.unwrap_or(hidden / heads),
        vocabulary_size: raw["vocab_size"].as_u64().unwrap(),
        rms_norm_eps: raw["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: raw["rope_theta"].as_f64().unwrap(),
        tie_word_embeddings: raw["tie_word_embeddings"].as_bool().unwrap(),
        attention_bias: raw["attention_bias"].as_bool().unwrap(),
        hidden_activation: raw["hidden_act"].as_str().map(str::to_string),
        rope_scaling: None,
        rope_layers,
    }
}

/// `HuggingFaceTB/SmolLM2-135M-Instruct`, whose layers all rotate.
fn smollm2_135m() -> Config {
    published(
        include_bytes!("fixtures/smollm2-135m-instruct.config.json"),
        None,
        None,
    )
}

/// `HuggingFaceTB/SmolLM3-3B`, with NoPE on every fourth layer.
fn smollm3_3b() -> Config {
    let raw = include_bytes!("fixtures/smollm3-3b.config.json");
    let plan = PositionPlan::from_huggingface_config(raw).unwrap();
    published(raw, Some(128), Some(plan.uses_rope().to_vec()))
}

fn segment(phase: Phase, sequence_length: u64, offset: u64, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length,
        position_offset: offset,
        start_layer: start,
        end_layer: end,
    }
}

fn check(name: &str, config: &Config, segment: Segment) {
    let mut sink = TextSink::default();
    build_graph(&mut sink, config, segment).unwrap();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, &sink.text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("golden graph {}: {error}", path.display()));
    assert_eq!(
        sink.text, expected,
        "graph '{name}' differs from tests/golden/{name}.txt"
    );
}

fn nope_in_the_middle() -> Option<Vec<bool>> {
    Some(vec![true, false, true])
}

#[test]
fn prefill_the_whole_two_layer_graph_with_every_layer_rotary() {
    check(
        "prefill_tiny_all_rope",
        &tiny(2, false, true, None),
        segment(Phase::Prefill, 3, 0, 0, 2),
    );
}

#[test]
fn decode_one_rotary_layer_carries_the_position_offset() {
    check(
        "decode_tiny_all_rope",
        &tiny(1, false, false, None),
        segment(Phase::Decode, 1, 11, 0, 1),
    );
}

#[test]
fn prefill_with_a_nope_layer() {
    check(
        "prefill_tiny_nope_layer",
        &tiny(3, false, true, nope_in_the_middle()),
        segment(Phase::Prefill, 3, 0, 0, 3),
    );
}

#[test]
fn decode_with_a_nope_layer() {
    check(
        "decode_tiny_nope_layer",
        &tiny(3, false, true, nope_in_the_middle()),
        segment(Phase::Decode, 1, 11, 0, 3),
    );
}

#[test]
fn prefill_with_a_nope_layer_and_attention_bias() {
    check(
        "prefill_tiny_nope_layer_with_bias",
        &tiny(3, true, false, nope_in_the_middle()),
        segment(Phase::Prefill, 3, 0, 0, 3),
    );
}

#[test]
fn decode_with_a_nope_layer_and_attention_bias() {
    check(
        "decode_tiny_nope_layer_with_bias",
        &tiny(3, true, false, nope_in_the_middle()),
        segment(Phase::Decode, 1, 11, 0, 3),
    );
}

#[test]
fn decode_middle_segment() {
    check(
        "decode_tiny_middle_segment",
        &tiny(3, false, true, nope_in_the_middle()),
        segment(Phase::Decode, 1, 11, 1, 2),
    );
}

#[test]
fn decode_last_segment_aliases_the_embedding() {
    check(
        "decode_tiny_last_segment_tied",
        &tiny(3, false, true, nope_in_the_middle()),
        segment(Phase::Decode, 1, 11, 2, 3),
    );
}

#[test]
fn prefill_with_banded_rope_scaling() {
    let mut config = tiny(1, false, false, None);
    config.rope_scaling = Some(RopeScaling::Banded(BandedRopeScaling {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 64,
    }));
    check(
        "prefill_tiny_banded_scaling",
        &config,
        segment(Phase::Prefill, 3, 0, 0, 1),
    );
}

#[test]
fn smollm2_135m_first_two_layers() {
    check(
        "smollm2_135m_prefill_layers_0_2",
        &smollm2_135m(),
        segment(Phase::Prefill, 5, 0, 0, 2),
    );
}

#[test]
fn smollm3_3b_segment_around_its_first_nope_layer() {
    check(
        "smollm3_3b_prefill_layers_2_5",
        &smollm3_3b(),
        segment(Phase::Prefill, 5, 0, 2, 5),
    );
}

#[test]
fn smollm3_3b_last_two_layers_end_on_a_nope_layer() {
    check(
        "smollm3_3b_decode_layers_34_36",
        &smollm3_3b(),
        segment(Phase::Decode, 1, 4096, 34, 36),
    );
}
