//! Three typed decision primitives and a fail-closed, ML-independent execution core.
//! Wire values are untrusted. Constructors and boundary checks validate them.
#![forbid(unsafe_code)]
pub mod types;
pub mod math;
pub mod schema;
pub mod sdk;
pub mod engine;
pub mod workflow;
pub mod storage;
pub mod demo;
pub use types::*;
