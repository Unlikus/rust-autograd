extern crate autograd as ag;
extern crate ndarray;
use ag::optimizers::adam;
use ag::variable::NamespaceTrait;
use ag::EvalError::OpError;
use ag::{EvalError, NdArray};
use ndarray::array;
use std::env::var;
use ag::tensor_ops as T;

type Tensor<'g> = ag::Tensor<'g, f32>;

#[test]
fn test_adam() {
    let mut ctx = ag::VariableEnvironment::<f32>::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    let w = ctx
        .default_namespace_mut()
        .slot()
        .set(rng.glorot_uniform(&[2, 2]));
    let b = ctx
        .default_namespace_mut()
        .slot()
        .set(ag::ndarray_ext::zeros(&[1, 2]));

    // Prepare adam optimizer
    let adam = adam::Adam::default(
        "my_unique_adam",
        ctx.default_namespace().current_var_ids(),
        &mut ctx, // mut env
    );

    println!(
        "default ns: current_var_names: {:?}",
        ctx.default_namespace().current_var_names()
    );
    println!(
        "default ns: current_var_ids: {:?}",
        ctx.default_namespace().current_var_ids()
    );

    println!(
        "adam ns: current_var_names: {:?}",
        ctx.namespace("my_unique_adam").current_var_names()
    );
    println!(
        "adam ns: current_var_ids: {:?}",
        ctx.namespace("my_unique_adam").current_var_ids()
    );

    ctx.run(|g| {
        let x = g.convert_to_tensor(array![[0.1, 0.2], [0.2, 0.1]]).show();
        let y = g.convert_to_tensor(array![1., 0.]).show();
        let w = g.variable_by_id(w);
        let b = g.variable_by_id(b);
        let z = T::matmul(x, w) + b;
        let loss = T::sparse_softmax_cross_entropy(z, &y);
        let mean_loss = T::reduce_mean(loss, &[0], false);
        let grads = &T::grad(&[&mean_loss], &[w, b]);
        let update_ops: &[Tensor] = &adam.update(&[w, b], grads, g);
        let updates = g.eval(update_ops, &[]);
        updates[0].as_ref().unwrap();
        updates[1].as_ref().unwrap();
    });
}

use autograd::ndarray::{Array, IxDyn};

#[test]
fn buggy() {
    let mut ctx = ag::VariableEnvironment::<f32>::new();
    let v = ctx.slot().set(ag::array_gen::zeros(&[]));

    let adam = adam::Adam::default(
        "my_unique_adam",
        ctx.default_namespace().current_var_ids(),
        &mut ctx, // mut env
    );

    ctx.run(|graph| {
        let c = graph.convert_to_tensor(ag::array_gen::ones(&[2]));
        let v = graph.variable_by_id(v);

        let y = c * v;
        let grads = T::grad(&[y], &[v]);

        let updates = adam.update(&[v], &grads, graph);
        for a in updates {
            a.show().eval(&[], graph);
        }
    })
}
