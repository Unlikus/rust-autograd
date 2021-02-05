extern crate autograd as ag;
extern crate ndarray;

use ag::optimizers::adam;
use ag::rand::seq::SliceRandom;
use ag::tensor_ops as T;
use ag::variable::NamespaceTrait;
use ag::{ndarray_ext as array, Graph};
use ndarray::s;
use std::collections::HashMap;
use std::time::Instant;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

mod mnist_data;

// This is a toy convolutional network for MNIST.
// Got 0.987 test accuracy in 100 sec on 2.7GHz Intel Core i5.
//
// First, run "./download_mnist.sh" beforehand if you don't have dataset and then run
// "cargo run --example cnn_mnist --release --features mkl" in `examples` directory.
macro_rules! timeit {
    ($x:expr) => {{
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}

fn conv_pool<'g>(x: Tensor<'g>, w: Tensor<'g>, b: Tensor<'g>) -> Tensor<'g> {
    let y1 = T::conv2d(x, w, 1, 1) + b;
    let y2 = T::relu(y1);
    T::max_pool2d(y2, 2, 0, 2)
}

fn inputs<'g>(g: &'g Graph<f32>) -> (Tensor<'g>, Tensor<'g>) {
    let x = g.placeholder(&[-1, 1, 28, 28]);
    let y = g.placeholder(&[-1, 1]);
    (x, y)
}

fn logits<'g>(x: Tensor<'g>, var: &HashMap<&str, Tensor<'g>>) -> Tensor<'g> {
    let z1 = conv_pool(x, var["w1"], var["b1"]); // map to 32 channel
    let z2 = conv_pool(z1, var["w2"], var["b2"]); // map to 64 channel
    let z3 = T::reshape(z2, &[-1, 64 * 7 * 7]); // flatten
    T::matmul(z3, var["w3"]) + var["b3"]
}

fn get_permutation(size: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..size).collect();
    perm.shuffle(&mut rand::thread_rng());
    perm
}

fn main() {
    // Get training data
    let ((x_train, y_train), (x_test, y_test)) = mnist_data::load();

    let max_epoch = 5;
    let batch_size = 200isize;
    let num_train_samples = x_train.shape()[0];
    let num_test_samples = x_test.shape()[0];
    let num_batches = num_train_samples / batch_size as usize;

    // Create training data
    let (x_train, x_test) = (
        x_train
            .into_shape(ndarray::IxDyn(&[num_train_samples, 1, 28, 28]))
            .unwrap(),
        x_test
            .into_shape(ndarray::IxDyn(&[num_test_samples, 1, 28, 28]))
            .unwrap(),
    );

    // Create trainable variables

    let mut env = ag::VariableEnvironment::<f32>::new();
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    env.slot()
        .with_name("w1")
        .set(rng.random_normal(&[32, 1, 3, 3], 0., 0.1));

    env.slot()
        .with_name("w2")
        .set(rng.random_normal(&[64, 32, 3, 3], 0., 0.1));

    env.slot()
        .with_name("w3")
        .set(rng.glorot_uniform(&[64 * 7 * 7, 10]));

    env.slot()
        .with_name("b1")
        .set(array::zeros(&[1, 32, 28, 28]));

    env.slot()
        .with_name("b2")
        .set(array::zeros(&[1, 64, 14, 14]));

    env.slot().with_name("b3").set(array::zeros(&[1, 10]));

    // Prepare adam optimizer
    let adam = adam::Adam::default(
        "my_unique_adam",
        env.default_namespace().current_var_ids(),
        &mut env, // mut env
    );

    // Training loop
    for epoch in 0..max_epoch {
        timeit!({
            for i in get_permutation(num_batches) {
                let i = i as isize * batch_size;
                let x_batch = x_train.slice(s![i..i + batch_size, .., .., ..]).into_dyn();
                let y_batch = y_train.slice(s![i..i + batch_size, ..]).into_dyn();

                env.run(|g| {
                    // make graph
                    let ns = g.env().default_namespace();
                    let var = g.var_tensors_by_name(&ns).collect::<HashMap<_, _>>();
                    let (x, y) = inputs(g);
                    let logits = logits(x, &var);
                    let loss = T::sparse_softmax_cross_entropy(&logits, &y);
                    let var_list: Vec<&Tensor> = var.values().collect();
                    let grads = &T::grad(&[&loss], &var_list);
                    let update_ops: &[Tensor] = &adam.update(&var_list, grads, g);

                    for result in g.eval(update_ops, &[x.given(x_batch), y.given(y_batch)]) {
                        result.unwrap();
                    }
                });
            }
            println!("finish epoch {}", epoch);
        });
    }

    // -- test --
    env.run(|g| {
        let ns = g.env().default_namespace();
        let var = g.var_tensors_by_name(&ns).collect::<HashMap<_, _>>();
        let (x, y) = inputs(g);
        let logits = logits(x, &var);
        let predictions = T::argmax(logits, -1, true);
        let accuracy = T::reduce_mean(&T::equal(predictions, &y), &[0, 1], false);
        println!(
            "test accuracy: {:?}",
            accuracy.eval(&[x.given(x_test.view()), y.given(y_test.view())], g)
        );
    })
}
