//! Print root-graph operator counts without adding an ONNX Python dependency.
use anyhow::{Context, Result};
use gigastt_quantize::onnx_proto::ModelProto;
use prost::Message;
use std::collections::BTreeMap;
fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("MODEL_ONNX")?;
    let model = ModelProto::decode(std::fs::read(path)?.as_slice())?;
    let graph = model.graph.context("ONNX model has no graph")?;
    let mut counts = BTreeMap::new();
    for node in graph.node {
        *counts.entry(node.op_type().to_owned()).or_insert(0usize) += 1;
    }
    for (op, count) in counts {
        println!("{op}\t{count}");
    }
    Ok(())
}
