//! Exact ONNX outlet metadata and runtime cache shapes of the qualified dynamic streaming graphs.
use super::StreamingTensorSpec;
use crate::model::TensorElement::{F32, I64};

/// Fixed physical front cache shapes, in manifest order.
pub const V4_STREAMING_FRONT_CACHE_SHAPES: &[&[usize]] = &[
    &[1, 1, 2, 64],
    &[1, 16, 1, 64],
    &[1, 32, 1, 31],
    &[1, 64, 2, 15],
];

/// Fixed physical layer cache shapes, in manifest order.
pub const V4_STREAMING_LAYER_CACHE_SHAPES: &[&[usize]] = &[
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 16, 1024],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 32, 768],
    &[1, 768, 2],
];

pub(super) const FRONTEND_INPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "features",
        element: F32,
        shape: &[1, 64, -1],
    },
    StreamingTensorSpec {
        name: "feature_lengths",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "front_0",
        element: F32,
        shape: &[1, 1, 2, 64],
    },
    StreamingTensorSpec {
        name: "front_1",
        element: F32,
        shape: &[1, 16, 1, 64],
    },
    StreamingTensorSpec {
        name: "front_2",
        element: F32,
        shape: &[1, 32, 1, 31],
    },
    StreamingTensorSpec {
        name: "front_3",
        element: F32,
        shape: &[1, 64, 2, 15],
    },
];

pub(super) const FRONTEND_OUTPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "pre_encoded",
        element: F32,
        shape: &[-1, -1, 768],
    },
    StreamingTensorSpec {
        name: "encoded_lengths",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "next_front_0",
        element: F32,
        shape: &[-1, -1, -1, -1],
    },
    StreamingTensorSpec {
        name: "next_front_1",
        element: F32,
        shape: &[-1, -1, -1, -1],
    },
    StreamingTensorSpec {
        name: "next_front_2",
        element: F32,
        shape: &[-1, -1, -1, -1],
    },
    StreamingTensorSpec {
        name: "next_front_3",
        element: F32,
        shape: &[-1, -1, -1, -1],
    },
];

pub(super) const STEP_INPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "features",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "feature_lengths",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "context_frames",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "position",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "cache_valid",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "cache_0",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_1",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_2",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_3",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_4",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_5",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_6",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_7",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_8",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_9",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_10",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_11",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_12",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_13",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_14",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_15",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_16",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_17",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_18",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_19",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_20",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_21",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_22",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_23",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_24",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_25",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_26",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_27",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_28",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_29",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_30",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_31",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_32",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_33",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_34",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_35",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_36",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_37",
        element: F32,
        shape: &[1, 16, 1024],
    },
    StreamingTensorSpec {
        name: "cache_38",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_39",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_40",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_41",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_42",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_43",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_44",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_45",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_46",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_47",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_48",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_49",
        element: F32,
        shape: &[1, 768, 2],
    },
    StreamingTensorSpec {
        name: "cache_50",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_51",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_52",
        element: F32,
        shape: &[1, 32, 768],
    },
    StreamingTensorSpec {
        name: "cache_53",
        element: F32,
        shape: &[1, 768, 2],
    },
];

pub(super) const STEP_OUTPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "encoded",
        element: F32,
        shape: &[-1, -1, -1],
    },
    StreamingTensorSpec {
        name: "encoded_lengths",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "next_position",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "next_cache_valid",
        element: I64,
        shape: &[1],
    },
    StreamingTensorSpec {
        name: "next_cache_0",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_1",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_2",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_3",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_4",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_5",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_6",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_7",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_8",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_9",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_10",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_11",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_12",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_13",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_14",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_15",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_16",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_17",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_18",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_19",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_20",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_21",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_22",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_23",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_24",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_25",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_26",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_27",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_28",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_29",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_30",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_31",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_32",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_33",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_34",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_35",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_36",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_37",
        element: F32,
        shape: &[1, -1, 1024],
    },
    StreamingTensorSpec {
        name: "next_cache_38",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_39",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_40",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_41",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_42",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_43",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_44",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_45",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_46",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_47",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_48",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_49",
        element: F32,
        shape: &[1, 768, -1],
    },
    StreamingTensorSpec {
        name: "next_cache_50",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_51",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_52",
        element: F32,
        shape: &[1, -1, 768],
    },
    StreamingTensorSpec {
        name: "next_cache_53",
        element: F32,
        shape: &[1, 768, -1],
    },
];

pub(super) const DECODER_INPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "x",
        element: I64,
        shape: &[-1, 1],
    },
    StreamingTensorSpec {
        name: "hi",
        element: F32,
        shape: &[1, -1, 320],
    },
    StreamingTensorSpec {
        name: "ci",
        element: F32,
        shape: &[1, -1, 320],
    },
];

pub(super) const DECODER_OUTPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "dec",
        element: F32,
        shape: &[-1, 1, 320],
    },
    StreamingTensorSpec {
        name: "ho",
        element: F32,
        shape: &[1, -1, 320],
    },
    StreamingTensorSpec {
        name: "co",
        element: F32,
        shape: &[1, -1, 320],
    },
];

pub(super) const JOINT_INPUTS: &[StreamingTensorSpec] = &[
    StreamingTensorSpec {
        name: "enc",
        element: F32,
        shape: &[-1, 768, -1],
    },
    StreamingTensorSpec {
        name: "dec",
        element: F32,
        shape: &[-1, 320, 1],
    },
];

pub(super) const JOINT_OUTPUTS: &[StreamingTensorSpec] = &[StreamingTensorSpec {
    name: "joint",
    element: F32,
    shape: &[-1, -1, 1, 1025],
}];
