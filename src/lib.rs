//! Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).
//!
//! ## Motivation
//! Machine learning is one of the field where Rust lagging behind other languages.
//! The aim of this crate is to show that Rust has the capability to imprement efficient and full-featured dataflow graph naturally.
//! Moreover, the core of this crate is quite small compared to others (due to being implemented in pure Rust and ndarray),
//! therefore it might be reasonable for those who are not familiar with how this kind of library works.
//!
//! ## Features
//! ### Lazy, lightweight tensor evaluation
//! Computation graphs are created on the fly (a.k.a. *define-by-run*), but are not evaluated until `eval` is called.
//! This mechanism balances better performance and flexibility.
//!
//! ```rust
//! use autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|g: &mut ag::Graph<_>| {
//!     let a: ag::Tensor<f32> = g.ones(&[60]);
//!     let b: ag::Tensor<f32> = g.ones(&[24]);
//!     let c: ag::Tensor<f32> = T::reshape(a, &[3, 4, 5]);
//!     let d: ag::Tensor<f32> = T::reshape(b, &[4, 3, 2]);
//!     let e: ag::Tensor<f32> = T::tensordot(c, d, &[1, 0], &[0, 1]);
//!     let result: ag::ndarray::Array<_, _> = e.eval(&[], g).unwrap();  // Getting `ndarray::Array` here.
//! });
//! ```
//!
//! ### Reverse-mode automatic differentiation
//! There are a lot of [built-in operations](https://docs.rs/autograd/1.0.0/autograd/struct.Graph.html)
//! that support *higher-order* derivatives, and
//! you can also [define your own differentiable ops](https://docs.rs/autograd/1.0.0/autograd/op/trait.Op.html) with ndarrays easily.
//!
//! Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
//!
//! ```rust
//! use autograd as ag;
//! use ag::tensor_ops as T;
//!
//! # fn main() {
//! ag::run(|g: &mut ag::Graph<_>| {
//!     let x = g.placeholder(&[]);
//!     let y = g.placeholder(&[]);
//!     let z = 2.*x*x + 3.*y + 1.;
//!
//!     // dz/dy
//!     let gy = &T::grad(&[z], &[y])[0];
//!     println!("{:?}", gy.eval(&[], g));   // => Ok(3.)
//!
//!     // dz/dx (requires to fill the placeholder `x`)
//!     let gx = &T::grad(&[z], &[x])[0];
//!     let feed = ag::ndarray::arr0(2.);
//!     println!("{:?}", gx.eval(&[x.given(feed.view())], g));  // => Ok(8.)
//!     // ddz/dx (differentiates `z` again)
//!     let ggx = &T::grad(&[gx], &[x])[0];
//!     println!("{:?}", ggx.eval(&[], g));  // => Ok(4.)
//! });
//! # }
//! ```
//!
//! ### Neural networks
//! This crate has various low-level features inspired by tensorflow/theano to train neural networks.
//! Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
//! ```rust
//! // This is a softmax regression for MNIST digits classification with Adam.
//! // This achieves 0.918 test accuracy after 3 epochs (0.11 sec/epoch on 2.7GHz Intel Core i5).
//! use autograd::{self as ag, optimizers::adam::Adam, variable::NamespaceTrait};
//! use autograd::tensor_ops::*;
//! use std::collections::HashMap;
//!
//! let mut env = ag::VariableEnvironment::new();
//!
//! let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
//!
//! // Register variables in this env.
//! // `with_name(name)` is optional but enables variable lookup using the name.
//! let w = env.slot().with_name("w").set(rng.glorot_uniform(&[28 * 28, 10]));
//! let b = env.slot().with_name("b").set(ag::ndarray_ext::zeros(&[1, 10]));
//!
//! let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
//!
//! let max_epoch = 3;
//!
//! for epoch in 0..max_epoch {
//!     env.run(|g| {
//!         let ns = g.env().default_namespace();
//!         let var = g.var_tensors_by_name(&ns).collect::<HashMap<_, _>>();
//!         let x = g.placeholder(&[-1, 28*28]);
//!         let y = g.placeholder(&[-1]);
//!         let z = matmul(x, var["w"]) + var["b"];
//!         let mean_loss = reduce_mean(sparse_softmax_cross_entropy(z, &y), &[0], false);
//!         let grads = &grad(&[&mean_loss], &[var["w"], var["b"]]);
//!         let updates: &[ag::Tensor<f32>] =
//!             &adam.update(&[var["w"], var["b"]], grads, g);
//!
//!         // let batch_size = 200isize;
//!         // let num_samples = x_train.shape()[0];
//!         // let num_batches = num_samples / batch_size as usize;
//!         // for i in get_permutation(num_batches) {
//!         //     let i = i as isize * batch_size;
//!         //     let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
//!         //     let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
//!         //     g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
//!         // }
//!     });
//! }
//! ```
//!
//! ### Hooks
//! You can register hooks on `ag::Tensor` objects for debugging.
//!
//! ```rust
//! use autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|g| {
//!     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
//!     let b: ag::Tensor<f32> = g.ones(&[2, 3]).show_shape();
//!     let c = T::matmul(a, b).show_with("MatMul:");
//!
//!     c.eval(&[], g);
//!     // [[0.0, 0.0],
//!     // [0.0, 0.0],
//!     // [0.0, 0.0],
//!     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
//!     //
//!     // [2, 3]
//!     //
//!     // MatMul:
//!     //  [[0.0, 0.0, 0.0],
//!     //  [0.0, 0.0, 0.0],
//!     //  [0.0, 0.0, 0.0],
//!     //  [0.0, 0.0, 0.0]] shape=[4, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2
//! });
//! ```
//!

#![feature(member_constraints)]
#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
/// re-exported for convenience and version-compatibility
pub extern crate ndarray;

// For intel MKL
#[cfg(all(feature = "blas", feature = "intel-mkl"))]
extern crate intel_mkl_src;
// #[cfg(all(feature = "blas", feature = "intel-mkl"))]
// extern crate intel_mkl_sys;

// For other blas impl
#[cfg(all(feature = "blas", not(feature = "intel-mkl")))]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate cblas_sys;

extern crate libc;
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
/// re-exported for convenience and version-compatibility
pub extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate rustc_hash;
pub(crate) extern crate smallvec;
extern crate uuid;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate approx;

mod gradient;
pub(crate) mod graph;
mod hook;
pub mod ndarray_ext;
pub mod op;
pub mod tensor_ops;
pub mod variable;
pub mod optimizers;
mod runtime;
pub mod tensor;
pub mod test_helper;
pub use variable::VariableEnvironment;

use rustc_hash::{FxHashMap, FxHashSet};
use std::any::TypeId;
use std::fmt;

/// Primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + Serialize
    + Deserialize<'static>
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + Serialize
    + Deserialize<'static>
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + Serialize
        + Deserialize<'static>
        + 'static
{
}

impl<T> Int for T where
    T: num::Integer
        + num_traits::NumAssignOps
        + num_traits::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + Sized
        + Serialize
        + Deserialize<'static>
        + 'static
{
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub(crate) fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::runtime::{Eval, Feed};

pub use crate::tensor::Tensor;

pub(crate) use crate::ndarray_ext::ArrRepr;

pub use crate::graph::{run, with, Graph, GraphRepr};
use serde::{Serialize, Deserialize};

/// Error during tensor's evaluation.
#[derive(Debug, PartialEq)]
pub enum EvalError {
    /// Error during `Op`'s computation.
    OpError(op::OpError),
}

impl std::error::Error for EvalError {}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvalError::OpError(e) => e.fmt(f),
        }
    }
}
