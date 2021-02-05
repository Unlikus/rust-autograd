# autograd

[![Build Status](https://travis-ci.org/raskr/rust-autograd.svg?branch=master)](https://travis-ci.org/raskr/rust-autograd)
[![Crates.io version](http://meritbadge.herokuapp.com/autograd)](https://crates.io/crates/autograd)
[![docs.rs](https://docs.rs/autograd/badge.svg)](https://docs.rs/autograd/)

Differentiable operations and tensors backed by [ndarray](https://github.com/rust-ndarray/ndarray).

## Motivation
Machine learning is one of the field where Rust lagging behind other languages.
The aim of this crate is to show that Rust has the capability to implement efficient and full-featured dataflow graph naturally.
Moreover, the core of this crate is quite small compared to others (due to being implemented in pure Rust and ndarray),
therefore it might be reasonable for those who are not familiar with how this kind of library works.

## Basic usage
``` toml
[dependencies]
autograd = "???"
```

## Enabling blas
If you use basic linalg operations, especially matrix multiplications, `blas` feature would be important to speed them up. 
``` toml
[dependencies]
autograd = {"???", features = ["blas", "<blas-implementation-choise>"] }
```

`<blas-implementation-choise>` must be one of the following (See also [blas-src](https://github.com/blas-lapack-rs/blas-src))
- `accelerate` macOS only
- `blis`
- `intel-mkl` Intel/AMD CPU only. Includes Vector Mathematics (VM) ops
- `netlib`
- `openblas`

## Features
### Lazy, lightweight tensor evaluation
Computation graphs are created on the fly (a.k.a. *define-by-run*), but are not evaluated until `eval` is called.
This mechanism balances better performance and flexibility.
```rust
use autograd as ag;
use autograd::tensor_ops::*;

ag::run(|g: &mut ag::Graph<_>| {
    let a: ag::Tensor<f32> = g.ones(&[60]);
    let b: ag::Tensor<f32> = g.ones(&[24]);
    let c: ag::Tensor<f32> = reshape(a, &[3, 4, 5]);
    let d: ag::Tensor<f32> = reshape(b, &[4, 3, 2]);
    let e: ag::Tensor<f32> = tensordot(c, d, &[1, 0], &[0, 1]);
    let result: ag::ndarray::Array<_, _> = e.eval(&[], g).unwrap();  // Getting `ndarray::Array` here.
});
```

### Reverse-mode automatic differentiation
There are a lot of [built-in operations](https://docs.rs/autograd/???/autograd/struct.Graph.html)
that support *higher-order* derivatives, and
you can also [define your own differentiable ops](https://docs.rs/autograd/???/autograd/op/trait.Op.html) with ndarrays easily.

Here we are just computing partial derivatives of `z = 2x^2 + 3y + 1`.
 ```rust
use autograd as ag;
use autograd::tensor_ops::*;

ag::run(|g: &mut ag::Graph<_>| {
    let x = g.placeholder(&[]);
    let y = g.placeholder(&[]);
    let z = 2.*x*x + 3.*y + 1.;

    // dz/dy
    let gy = &g.grad(&[z], &[y])[0];
    println!("{:?}", gy.eval(&[], g));   // => Ok(3.)

    // dz/dx (requires to fill the placeholder `x`)
    let gx = &grad(&[z], &[x])[0];
    let feed = ag::ndarray::arr0(2.);
    println!("{:?}", gx.eval(&[x.given(feed.view())], g));  // => Ok(8.)
    // ddz/dx (differentiates `z` again)
    let ggx = &grad(&[gx], &[x])[0];
    println!("{:?}", ggx.eval(&[], g));  // => Ok(4.)
});
 ```

 ### Neural networks
 This crate has various low-level features inspired by tensorflow/theano to train neural networks.
 Since computation graphs require only bare minimum of heap allocations, the overhead is small, even for complex networks.
 ```rust
// This is a softmax regression for MNIST digits classification with Adam.
// This achieves 0.918 test accuracy after 3 epochs (0.11 sec/epoch on 2.7GHz Intel Core i5).
use autograd::{self as ag, optimizers::adam::Adam, variable::NamespaceTrait};
use autograd::tensor_ops::*;
use std::collections::HashMap;

let mut env = ag::VariableEnvironment::new();

let rng = ag::ndarray_ext::ArrayRng::<f32>::default();

// Register variables in this env.
// `with_name(name)` is optional but enables variable lookup using the name.
let w = env.slot().with_name("w").set(rng.glorot_uniform(&[28 * 28, 10]));
let b = env.slot().with_name("b").set(ag::ndarray_ext::zeros(&[1, 10]));

let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);

let max_epoch = 3;

for epoch in 0..max_epoch {
    env.run(|g| {
        let ns = g.env().default_namespace();
        let var = g.var_tensors_by_name(&ns).collect::<HashMap<_, _>>();
        let x = g.placeholder(&[-1, 28*28]);
        let y = g.placeholder(&[-1]);
        let z = matmul(x, var["w"]) + var["b"];
        let mean_loss = reduce_mean(sparse_softmax_cross_entropy(z, &y), &[0], false);
        let grads = &grad(&[&mean_loss], &[var["w"], var["b"]]);
        let updates: &[ag::Tensor<f32>] =
            &adam.update(&[var["w"], var["b"]], grads, g);

        // let batch_size = 200isize;
        // let num_samples = x_train.shape()[0];
        // let num_batches = num_samples / batch_size as usize;
        // for i in get_permutation(num_batches) {
        //     let i = i as isize * batch_size;
        //     let x_batch = x_train.slice(s![i..i + batch_size, ..]).into_dyn();
        //     let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();
        //     g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]);
        // }
    });
}
 ```

 ConvNet, LSTM example can be found in [examples](https://github.com/raskr/rust-autograd/tree/master/examples)

 ### Hooks
 You can register hooks on `ag::Tensor` objects for debugging.
 ```rust
use autograd as ag;

ag::run(|g| {
    let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
    let b: ag::Tensor<f32> = g.ones(&[2, 3]).show_shape();
    let c = g.matmul(a, b).show_with("MatMul:");

    c.eval(&[], g);
    // [[0.0, 0.0],
    // [0.0, 0.0],
    // [0.0, 0.0],
    // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    //
    // [2, 3]
    //
    // MatMul:
    //  [[0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0],
    //  [0.0, 0.0, 0.0]] shape=[4, 3], strides=[3, 1], layout=C (0x1), dynamic ndim=2
});
 ```

For more, see [documentation](https://docs.rs/autograd/) or
[examples](https://github.com/raskr/rust-autograd/tree/master/examples)
