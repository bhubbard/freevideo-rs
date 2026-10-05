//! Safetensors weight management, block decomposition, and interleaved memory residency.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Spread any affordable prefix of pinned layers across the computation.
/// Uses longest-gap bisection to ensure even coverage across the model blocks.
pub fn interleaved_layer_order(count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    // Max-heap on (gap, Reverse(start), end)
    let mut heap = BinaryHeap::new();
    heap.push((count, Reverse(0usize), count));

    let mut order = Vec::with_capacity(count);
    while let Some((_, Reverse(start), end)) = heap.pop() {
        let middle = (start + end) / 2;
        order.push(middle);

        if start < middle {
            heap.push((middle - start, Reverse(start), middle));
        }
        if middle + 1 < end {
            heap.push((end - (middle + 1), Reverse(middle + 1), end));
        }
    }
    order
}

/// Identifies the caching group for a given tensor.
/// e.g. "transformer_blocks.0.attn.q.weight" -> "blocks/00"
///      "transformer_blocks.1.adaln_proj.weight" -> "adaln/01"
///      "pos_embed" -> "root"
pub fn group_name(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.is_empty() || parts[0] != "transformer_blocks" || parts.len() < 3 {
        return "root".to_string();
    }
    let block_idx: usize = parts[1].parse().unwrap_or(0);
    let prefix = if parts[2] == "adaln_proj" {
        "adaln/"
    } else {
        "blocks/"
    };
    format!("{}{:02}", prefix, block_idx)
}

pub fn sha256_file<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub group: String,
    pub byte_size: usize,
}

pub fn inspect_safetensors<P: AsRef<Path>>(path: P) -> io::Result<Vec<TensorMeta>> {
    let mut file = File::open(path)?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let header_str =
        String::from_utf8(header_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let json_val: serde_json::Value = serde_json::from_str(&header_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut tensors = Vec::new();
    if let serde_json::Value::Object(map) = json_val {
        for (name, val) in map {
            if name == "__metadata__" {
                continue;
            }
            let shape = val
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let dtype = val
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("F32")
                .to_string();
            let offsets = val
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let byte_size = if offsets.len() == 2 && offsets[1] >= offsets[0] {
                offsets[1] - offsets[0]
            } else {
                0
            };

            let group = group_name(&name);
            tensors.push(TensorMeta {
                name,
                shape,
                dtype,
                group,
                byte_size,
            });
        }
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaved_layer_order() {
        let order4 = interleaved_layer_order(4);
        assert_eq!(order4, vec![2, 1, 0, 3]);

        let order8 = interleaved_layer_order(8);
        assert_eq!(order8.len(), 8);
        // All indices 0..8 should be present exactly once
        let mut sorted = order8.clone();
        sorted.sort();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_group_name() {
        assert_eq!(
            group_name("transformer_blocks.0.attn.q.weight"),
            "blocks/00"
        );
        assert_eq!(
            group_name("transformer_blocks.12.adaln_proj.bias"),
            "adaln/12"
        );
        assert_eq!(group_name("time_embed.weight"), "root");
    }
}
