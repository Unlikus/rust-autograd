//! Stochastic gradient descent optimizer
use crate::tensor_ops::gradient_descent_ops::sgd;
use crate::tensor::Tensor;
use crate::{Float, Graph};

/// Vanilla SGD optimizer
///
/// ```
/// extern crate autograd as ag;
///
/// use ag::optimizers::adam;
/// use ag::variable::NamespaceTrait;
///
/// // Define parameters to optimize.
/// let mut env = ag::VariableEnvironment::new();
/// let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// let w = env.slot().set(rng.glorot_uniform(&[28 * 28, 10]));
/// let b = env.slot().set(ag::ndarray_ext::zeros(&[1, 10]));
///
/// let sgd = ag::optimizers::sgd::SGD { lr: 0.1 };
///
/// env.run(|g| {
///     let w = g.variable_by_id(w);
///     let b = g.variable_by_id(b);
///
///     // some operations using w and b
///     // let y = ...
///     // let grads = g.grad(&[y], &[w, b]);
///
///     // Getting update ops of `params` using its gradients and adam.
///     // let updates: &[ag::Tensor<f32>] = &sgd.update(&[w, b], &grads, &g);
///
///     // for result in &g.eval(updates, &[]) {
///     //     println!("updates: {:?}", result.unwrap());
///     // }
/// });
/// ```
///
/// See also <https://github.com/raskr/rust-autograd/blob/master/examples/mlp_mnist.rs>.
pub struct SGD<T: Float> {
    /// Learning rate
    pub lr: T,
}

impl<'g, T: Float> SGD<T> {
    /// Updates `params` with SGD.
    ///
    /// Returns the updates for `params`.
    pub fn update(
        &self,
        params: &[Tensor<'g, T>],
        grads: &[Tensor<'g, T>],
        g: &'g Graph<T>,
    ) -> Vec<Tensor<'g, T>> {
        let len = params.len();
        let mut ret = Vec::with_capacity(len);
        for i in 0..len {
            ret.push(
                Tensor::builder(g)
                    .append_input(&params[i], true)
                    .append_input(&grads[i], false)
                    .build(sgd::SGDOp::new(self.lr)),
            );
        }
        ret
    }
}
