//! Adam optimizer
use crate::tensor_ops::gradient_descent_ops::adam;
use crate::tensor::Tensor;
use crate::variable::VarArrayID;
use crate::{Float, Graph, VariableEnvironment};

/// Adam optimizer
///
/// This implementation is based on <http://arxiv.org/abs/1412.6980v8>.
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
/// // Adam optimizer with default params.
/// // State arrays are created in the "my_adam" namespace.
/// let adam = adam::Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
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
///     // let updates: &[ag::Tensor<f32>] = &adam.update(&[w, b], &grads, &g);
///
///     // for result in &g.eval(updates, &[]) {
///     //     println!("updates: {:?}", result.unwrap());
///     // }
/// });
/// ```
///
/// See also <https://github.com/raskr/rust-autograd/blob/master/examples/>
pub struct Adam<F: Float> {
    static_params: StaticParams<F>,
    adam_namespace_id: &'static str,
}

impl<'t, 'g, 's, F: Float> Adam<F> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    pub fn default(
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item =VarArrayID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> Adam<F> {
        let static_params = StaticParams {
            alpha: F::from(0.001).unwrap(),
            eps: F::from(1e-08).unwrap(),
            b1: F::from(0.9).unwrap(),
            b2: F::from(0.999).unwrap(),
        };
        Adam::new(
            static_params,
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Instantiates `Adam` optimizer with given params.
    pub fn new(
        static_params: StaticParams<F>,
        var_id_list: impl IntoIterator<Item =VarArrayID>,
        env: &mut VariableEnvironment<F>,
        adam_namespace_id: &'static str,
    ) -> Adam<F> {
        for vid in var_id_list.into_iter() {
            let m_name = format!("{}m", vid);
            let v_name = format!("{}v", vid);
            let t_name = format!("{}t", vid);
            let (m, v, t) = {
                let target_var = env.get_array_by_id(vid).expect("variable array not found").borrow();
                let var_shape = target_var.shape();
                (
                    crate::ndarray_ext::zeros(var_shape),
                    crate::ndarray_ext::zeros(var_shape),
                    crate::ndarray_ext::from_scalar(F::one()),
                )
            };
            let mut adam_ns = env.namespace_mut(adam_namespace_id);
            adam_ns.slot().with_name(m_name).set(m);
            adam_ns.slot().with_name(v_name).set(v);
            adam_ns.slot().with_name(t_name).set(t);
        }
        Adam {
            static_params,
            adam_namespace_id,
        }
    }

    /// Creates ops to optimize `params` with Adam.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn update<A, B>(&self, params: &[A], grads: &[B], g: &'g Graph<F>) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        let num_params = params.len();
        assert_eq!(num_params, grads.len());
        let mut ret = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let param = params[i].as_ref();
            let namespace = g.env().namespace(&self.adam_namespace_id);
            let var_id = param.get_variable_id().expect("Got non-variable tensor");
            let m = g.variable_by_name(&format!("{}m", var_id), &namespace);
            let v = g.variable_by_name(&format!("{}v", var_id), &namespace);
            let t = g.variable_by_name(&format!("{}t", var_id), &namespace);

            ret.push(
                Tensor::builder(g)
                    .append_input(param, true)
                    .append_input(grads[i].as_ref(), false)
                    .append_input(&m, true)
                    .append_input(&v, true)
                    .append_input(&t, true)
                    .build(
                        adam::AdamOp {
                            static_params: self.static_params.clone(),
                        },
                    ),
            );
        }
        ret
    }
}

/// Holds Adam's static parameters (`alpha`, `eps`, `b1`, `b2`).
#[derive(Clone)]
pub struct StaticParams<T: Float> {
    pub alpha: T,
    pub eps: T,
    pub b1: T,
    pub b2: T,
}
