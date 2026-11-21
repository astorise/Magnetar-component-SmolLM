//! SmolLM model component for Magnetar.
//!
//! This initial implementation reads SmolLM3's per-layer positional
//! configuration, and provides the Llama-compatible graph producer used by
//! SmolLM2. SmolLM3 graph production and inference are not implemented yet.

mod llama_graph;

use serde::Deserialize;
use std::error::Error;
use std::fmt;

/// An upper bound on layer counts accepted from an untrusted configuration.
const MAX_LAYERS: usize = 4096;

/// SmolLM3's positional encoding, in decoder-layer order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionPlan {
    uses_rope: Vec<bool>,
}

impl PositionPlan {
    /// Read the positional-encoding plan from a Hugging Face `config.json`.
    ///
    /// `no_rope_layers` is an unfortunately named mask: `1` enables RoPE
    /// and `0` disables it. An explicit mask takes precedence over the
    /// interval; an omitted or null mask defaults to one NoPE layer every
    /// four layers unless `no_rope_layer_interval` is provided.
    ///
    /// This validates only the fields needed for the position plan. A
    /// successful result does not imply that the checkpoint can be executed.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ConfigError> {
        let config: PositionConfig = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid SmolLM3 config: {error}")))?;
        if config.model_type != "smollm3" {
            return Err(ConfigError(format!(
                "expected model_type smollm3, got {:?}",
                config.model_type
            )));
        }
        let layers = config.num_hidden_layers;
        if !(1..=MAX_LAYERS).contains(&layers) {
            return Err(ConfigError(format!(
                "num_hidden_layers must be between 1 and {MAX_LAYERS}"
            )));
        }

        let uses_rope = if let Some(mask) = config.no_rope_layers {
            if mask.len() < layers {
                return Err(ConfigError(format!(
                    "no_rope_layers has {} entries, expected at least {layers}",
                    mask.len()
                )));
            }
            mask.into_iter()
                .take(layers)
                .enumerate()
                .map(|(layer, entry)| match entry {
                    0 => Ok(false),
                    1 => Ok(true),
                    _ => Err(ConfigError(format!(
                        "no_rope_layers[{layer}] must be 0 or 1"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let interval = config.no_rope_layer_interval.unwrap_or(4);
            if interval == 0 {
                return Err(ConfigError(
                    "no_rope_layer_interval must be nonzero".to_string(),
                ));
            }
            (0..layers)
                .map(|layer| (layer + 1) % interval != 0)
                .collect()
        };
        Ok(Self { uses_rope })
    }

    /// Whether each decoder layer uses RoPE; `false` denotes a NoPE layer.
    pub fn uses_rope(&self) -> &[bool] {
        &self.uses_rope
    }
}

#[derive(Deserialize)]
struct PositionConfig {
    model_type: String,
    num_hidden_layers: usize,
    no_rope_layers: Option<Vec<u8>>,
    no_rope_layer_interval: Option<usize>,
}

/// A malformed or unsupported positional configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ConfigError {}
