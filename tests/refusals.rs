//! A checkpoint or a segment this Component cannot serve is refused before any
//! call to the graph-builder: the Runtime is left with no half-built graph.

use magnetar_component_sdk::testing::RecordingSink;
use magnetar_component_smollm::plan::{Config, Phase, RopeScaling, Segment, build_graph};

fn config() -> Config {
    Config {
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dimension: 4,
        vocabulary_size: 17,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        attention_bias: false,
        hidden_activation: Some("silu".into()),
        rope_scaling: None,
        rope_layers: None,
    }
}

fn segment() -> Segment {
    Segment {
        phase: Phase::Prefill,
        sequence_length: 3,
        position_offset: 0,
        start_layer: 0,
        end_layer: 3,
    }
}

fn refused(config: &Config, segment: Segment) -> String {
    let mut sink = RecordingSink::new();
    let reason = build_graph(&mut sink, config, segment).unwrap_err();
    assert!(sink.began.is_empty(), "begin_graph was called");
    assert!(sink.nodes.is_empty(), "a node was added");
    assert!(sink.weights.is_empty(), "a weight edge was asked for");
    reason
}

#[test]
fn an_activation_other_than_silu_is_refused() {
    let mut config = config();
    config.hidden_activation = Some("gelu".into());
    assert_eq!(
        refused(&config, segment()),
        "SmolLM does not implement checkpoint activation \"gelu\""
    );
}

#[test]
fn an_unknown_activation_is_refused() {
    let mut config = config();
    config.hidden_activation = None;
    assert_eq!(
        refused(&config, segment()),
        "SmolLM graph production requires a known checkpoint activation"
    );
}

#[test]
fn yarn_frequency_scaling_is_refused() {
    let mut config = config();
    config.rope_scaling = Some(RopeScaling::Yarn);
    assert_eq!(
        refused(&config, segment()),
        "SmolLM2 does not support YaRN RoPE"
    );
}

#[test]
fn a_rope_selection_with_the_wrong_number_of_layers_is_refused() {
    let mut config = config();
    config.rope_layers = Some(vec![true, false]);
    assert_eq!(
        refused(&config, segment()),
        "invalid per-layer RoPE selection"
    );
}

#[test]
fn a_segment_that_is_not_a_range_of_the_layers_is_refused() {
    for (start_layer, end_layer) in [(2, 1), (0, 4), (4, 4)] {
        let segment = Segment {
            start_layer,
            end_layer,
            ..segment()
        };
        assert_eq!(refused(&config(), segment), "invalid layer segment");
    }
}

#[test]
fn a_position_offset_beyond_the_graph_limit_is_refused() {
    let segment = Segment {
        position_offset: i64::MAX as u64 + 1,
        ..segment()
    };
    // The reason carries the conversion error's `Debug` text, which is what
    // the `expect` this replaced printed.
    let reason = refused(&config(), segment);
    assert!(
        reason.starts_with("position offset exceeds graph limits: TryFromIntError("),
        "{reason}"
    );
}

#[test]
fn an_empty_segment_builds_the_input_and_no_layer() {
    let mut sink = RecordingSink::new();
    let segment = Segment {
        start_layer: 1,
        end_layer: 1,
        ..segment()
    };
    build_graph(&mut sink, &config(), segment).unwrap();
    assert_eq!(sink.inputs.len(), 1);
    assert!(sink.nodes.is_empty());
}
