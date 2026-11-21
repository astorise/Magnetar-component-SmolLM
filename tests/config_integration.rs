use magnetar_component_smollm::PositionPlan;

#[test]
fn published_smollm3_checkpoint_has_rope_on_three_of_every_four_layers() {
    let config = include_bytes!("fixtures/smollm3-3b.config.json");
    let plan = PositionPlan::from_huggingface_config(config).unwrap();
    assert_eq!(plan.uses_rope().len(), 36);
    for (index, &uses_rope) in plan.uses_rope().iter().enumerate() {
        assert_eq!(uses_rope, (index + 1) % 4 != 0, "layer {index}");
    }
}

#[test]
fn explicit_mask_takes_precedence_and_may_be_longer_than_the_layer_count() {
    let config = br#"{
        "model_type": "smollm3", "num_hidden_layers": 3,
        "no_rope_layers": [0, 1, 0, 1], "no_rope_layer_interval": 0
    }"#;
    assert_eq!(
        PositionPlan::from_huggingface_config(config)
            .unwrap()
            .uses_rope(),
        &[false, true, false]
    );
}

#[test]
fn missing_or_null_mask_uses_the_documented_default_and_custom_interval() {
    let default = br#"{"model_type":"smollm3","num_hidden_layers":5}"#;
    let custom = br#"{
        "model_type":"smollm3","num_hidden_layers":5,
        "no_rope_layers":null,"no_rope_layer_interval":2
    }"#;
    assert_eq!(
        PositionPlan::from_huggingface_config(default)
            .unwrap()
            .uses_rope(),
        &[true, true, true, false, true]
    );
    assert_eq!(
        PositionPlan::from_huggingface_config(custom)
            .unwrap()
            .uses_rope(),
        &[true, false, true, false, true]
    );
}

#[test]
fn rejects_invalid_configs_instead_of_silently_changing_position_semantics() {
    let cases: &[&[u8]] = &[
        br#"{"model_type":"llama","num_hidden_layers":2}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":0}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":4097}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":2,"no_rope_layers":[1]}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":2,"no_rope_layers":[1,2]}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":2,"no_rope_layers":[1,true]}"#,
        br#"{"model_type":"smollm3","num_hidden_layers":2,"no_rope_layer_interval":0}"#,
        b"not-json",
    ];
    for config in cases {
        assert!(
            PositionPlan::from_huggingface_config(config).is_err(),
            "invalid config accepted: {}",
            String::from_utf8_lossy(config)
        );
    }
}
