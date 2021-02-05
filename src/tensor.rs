//! Defining things related to `ag::Tensor`.
use crate::Float;
use crate::{op, Graph};
use crate::{NdArray, NdArrayView};

use crate::graph::{AsGraphRepr, GraphRepr};
use crate::op::{GradientContext, InputArray, OpError};
use crate::variable::{VariableEnvironment, VarArrayID};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::cell::Ref;

/// Lazy N-dimensional array.
///
/// `Tensor` is:
///
/// - created by operations of a `Graph`.
/// - not evaluated until `Tensor::eval`, `Graph::eval` or `Eval::run` is called.
/// - cheap to `Copy` since it contains only refs to the owned internal objects.
///
/// The builtin operations for tensors are provided as [Graph's methods](../graph/struct.Graph.html).
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops as T;
///
/// let mut env = ag::VariableEnvironment::new();
/// let var = env.slot().set(ag::ndarray_ext::ones(&[2, 3]));
///
/// env.run(|graph| {  // `Graph` is necessary to create tensors.
///     // `random` is just a symbolic object belongs to `graph`.
///     let random: ag::Tensor<f64> = graph.standard_normal(&[2, 3]);
///
///     // Getting the tensor associated with a pre-registered variable array.
///     let var = graph.variable_by_id(var);
///
///     // This is ok since tensor's binary operators are overloaded!
///     let mul = random * 3. + var;
///
///     // Evaluates the symbolic tensor as an ndarray::Array<T, IxDyn>.
///     type NdArray = ag::NdArray<f64>;
///     let mul_val: Result<NdArray, ag::EvalError> = mul.eval(&[], graph);
///
///     // Reshapes the tensor without copy (ArrayView is used internally).
///     let reshaped = T::reshape(random, &[6]);
///
///     // Evaluating multiple tensors at once.
///     // Note that although `random` node is required two times in this computation graph,
///     // it's evaluated only once since `eval()` is smart enough to avoid duplicated computations.
///     let pair: Vec<Result<NdArray, _>> = graph.eval(&[mul, reshaped], &[]);
/// });
/// ```
#[derive(Clone, Copy)]
pub struct Tensor<'graph, F: Float> {
    pub(crate) id: usize,
    pub(crate) graph: &'graph GraphRepr<F>,
}

impl<'graph, F: Float> Tensor<'graph, F> {
    pub(crate) fn input_tensors(&self) -> Ref<InputArray<Input>> {
        Ref::map(self.inner(), |x| &x.in_edges)
    }

    // Returns the i-th input node of this tensor
    pub(crate) fn input_tensor(
        &self,
        i: usize,
        g: &'graph GraphRepr<F>,
    ) -> Option<Tensor<'graph, F>> {
        self.inner().in_edges.get(i).map(|x| x.as_tensor(g))
    }

    #[inline]
    pub(crate) fn inner<'t>(&self) -> Ref<TensorInternal<F>> {
        self.graph.access_inner(self.id)
    }

    /// Returns the graph to which this tensor belongs.
    #[inline]
    pub fn graph(&self) -> &'graph GraphRepr<F> {
        self.graph
    }

    /// Returns a mutable ref of the graph to which this tensor belongs.
    #[inline]
    pub fn graph_mut(&mut self) -> &'graph GraphRepr<F> {
        &mut self.graph
    }

    /// Evaluates this tensor as an `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.zeros(&[2]);
    ///    assert_eq!(a.eval(&[], g), Ok(array![0., 0.].into_dyn()));
    /// });
    /// ```
    ///
    /// See also [Graph::eval](../graph/struct.Graph.html#method.eval).
    pub fn eval<'v>(
        &self,
        feeds: &'v [crate::runtime::Feed<'v, F>],
        graph: &Graph<F>,
    ) -> Result<NdArray<F>, crate::EvalError> {
        crate::graph::assert_same_graph(graph, self.graph);
        let mut ret = graph.eval(&[self], feeds);
        debug_assert_eq!(ret.len(), 1);
        ret.remove(0)
    }

    /// Retruns a `Feed` assigning a given value to this (placeholder) tensor.
    ///
    /// Ensure that the return value is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let x = g.placeholder(&[2]);
    ///
    ///     // Fills the placeholder with an ArrayView, then eval.
    ///     let value = array![1., 1.];
    ///     x.eval(&[
    ///         x.given(value.view())
    ///     ], g);
    /// });
    /// ```
    pub fn given<D>(self, value: ndarray::ArrayView<F, D>) -> crate::runtime::Feed<F>
    where
        D: ndarray::Dimension,
    {
        assert!(
            self.is_placeholder(),
            "Receiver of Tensor::given must be a placeholder."
        );
        self.inner().validate_feed_shape(value.shape());
        crate::runtime::Feed::new(self.id(), value.into_dyn())
    }

    #[inline]
    /// Creates a new [TensorBuilder](struct.TensorBuilder.html).
    pub fn builder(graph: &'graph impl AsGraphRepr<F>) -> TensorBuilder<'graph, F> {
        // Starts with default values
        TensorBuilder {
            graph: graph.as_graph_repr(),
            shape: None,
            in_nodes: op::InputArray::new(),
            can_have_gradient: true,
            is_placeholder: false,
            input_indices: None,
            backprop_inputs: None,
            known_shape: None,
            variable_id: None,
        }
    }

    // Registers a hook on the receiver tensor.
    //
    // ```
    // use autograd as ag;
    //
    // ag::with(|g| {
    //     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).register_hook(ag::hook::Show);
    //     let b: ag::Tensor<f32> = g.ones(&[2, 3]).register_hook(ag::hook::ShowShape);
    //     let c = g.matmul(a, b);
    //
    //     c.eval(&[]);
    //     // [[0.0, 0.0],
    //     // [0.0, 0.0],
    //     // [0.0, 0.0],
    //     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    //
    //     // [2, 3]
    // });
    // ```
    #[inline]
    fn register_hook<H: crate::hook::Hook<F> + 'static>(self, hook: H) -> Tensor<'graph, F> {
        Tensor::builder(self.graph)
            .append_input(&self, false)
            .build(crate::tensor_ops::hook_ops::HookOp::new(hook))
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stderr.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
    ///     a.eval(&[], g);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///     });
    /// ```
    #[inline]
    pub fn show(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Show)
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stderr, with given prefix.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show_with("My value:");
    ///     a.eval(&[], g);
    ///     // My value:
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    /// });
    ///
    /// ```
    #[inline]
    pub fn show_with(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowWith(what))
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stderr.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape();
    ///     a.eval(&[], g);
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowShape)
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stderr, with given prefix.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape_with("My shape:");
    ///     a.eval(&[], g);
    ///     // My shape:
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape_with(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowShapeWith(what))
    }

    /// Sets a hook that displays the given string after evaluation of the receiver tensor.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).print("This is `a`");
    ///     a.eval(&[], g);
    ///     // This is `a`
    /// });
    /// ```
    #[inline]
    pub fn print(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Print(what))
    }

    /// Sets a hook that calls the given closure after evaluation of the receiver tensor.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).raw_hook(|arr| println!("{:?}", arr));
    ///     a.eval(&[], g);
    /// });
    /// ```
    #[inline]
    pub fn raw_hook<FUN: Fn(&NdArrayView<F>) -> () + 'static + Send + Sync>(
        self,
        f: FUN,
    ) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Raw {
            raw: f,
            phantom: PhantomData,
        })
    }

    /// Returns the id of this tensor in this graph.
    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.inner().num_inputs()
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_backprop_inputs(&self) -> usize {
        let inner = self.inner();
        inner
            .backprop_inputs
            .as_ref()
            .unwrap_or(&inner.in_edges)
            .len()
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        self.inner().is_source()
    }

    #[inline]
    pub(crate) fn get_variable_id(&self) -> Option<VarArrayID> {
        self.inner().variable_id
    }

    #[inline]
    /// Input node used when backprop.
    pub fn get_backprop_input(&self, idx: usize) -> Tensor<'graph, F> {
        self.graph
            .tensor(self.inner().get_backprop_inputs()[idx].id)
    }

    #[inline]
    pub fn is_placeholder(&self) -> bool {
        self.inner().is_placeholder
    }

    #[inline]
    pub fn is_differentiable(&self) -> bool {
        self.inner().is_differentiable
    }

    /// True is this tensor was created by `Graph::variable`.
    #[inline]
    #[allow(unused)]
    pub(crate) fn is_variable(&self) -> bool {
        self.inner().is_variable()
    }
}

impl<'b, T: Float> AsRef<Tensor<'b, T>> for Tensor<'b, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<'b, T> {
        self
    }
}

pub(crate) struct TensorInternal<F: Float> {
    /// Tensor ID. Unique in the graph which this tensor belongs to.
    pub(crate) id: usize,

    /// Operation to evaluate this tensor.
    pub(crate) op: Option<Box<dyn op::Op<F>>>,

    /// References to immediate predecessors.
    pub(crate) in_edges: op::InputArray<Input>,

    /// The rank number for topological ordering in a graph.
    pub(crate) topo_rank: usize,

    /// *Symbolic* shape of this tensor.
    pub(crate) shape: Option<usize>,

    /// This tensor is placeholder or not.
    pub(crate) is_placeholder: bool,

    /// This is true if this tensor can have gradient for any objectives.
    pub(crate) is_differentiable: bool,

    /// Input indices of arrays used in `compute`
    pub(crate) input_indices: op::InputArray<usize>,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) backprop_inputs: Option<op::InputArray<Input>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub(crate) known_shape: Option<KnownShape>,

    /// ID to lookup variable array in VariableEnvironment
    pub(crate) variable_id: Option<VarArrayID>,
}

impl<F: Float> TensorInternal<F> {
    /// Returns the Op of this tensor
    pub fn get_op(&self) -> &Box<dyn op::Op<F>> {
        self.op
            .as_ref()
            .expect("bad impl: Op is now stolen in gradient.rs")
    }

    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_source(&self) -> bool {
        self.in_edges.is_empty()
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_variable(&self) -> bool {
        self.variable_id.is_some()
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub(crate) fn num_inputs(&self) -> usize {
        self.in_edges.len()
    }

    /// True if the op of this tensor is differentiable
    #[inline]
    #[allow(dead_code)]
    pub fn is_differentiable(&self) -> bool {
        self.is_differentiable
    }

    #[inline]
    pub(crate) fn validate_feed_shape(&self, shape: &[usize]) {
        debug_assert!(self.is_placeholder);
        if !self.known_shape.as_ref().unwrap().validate(shape) {
            panic!(
                "Shape error: placeholder required {:?}, but got {:?}",
                self.known_shape.as_ref().unwrap().get(),
                shape
            );
        }
    }

    #[inline]
    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) fn get_backprop_inputs(&self) -> &[Input] {
        self.backprop_inputs
            .as_ref()
            .unwrap_or(&self.in_edges)
            .as_slice()
    }
}

impl<T: Float> fmt::Debug for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Node name: {}, id: {}, num of inputs: {}, in-edges: {:?}",
            self.get_op().name(),
            self.id(),
            self.in_edges.len(),
            self.in_edges
        )
    }
}

// empty implementation
impl<T: Float> Eq for TensorInternal<T> {}

impl<T: Float> PartialEq for TensorInternal<T> {
    #[inline(always)]
    fn eq(&self, other: &TensorInternal<T>) -> bool {
        // compare addresses on the heap
        self.id() == other.id()
    }
}

/// Raw pointer hashing
impl<T: Float> Hash for TensorInternal<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Float> AsRef<TensorInternal<T>> for TensorInternal<T> {
    #[inline(always)]
    fn as_ref(&self) -> &TensorInternal<T> {
        self
    }
}

impl<T: Float> fmt::Display for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "name={}", self.get_op().name(),)
    }
}

/// A decorated `Tensor` passed to `TensorBuilder::set_inputs`.
///
/// Use `new` to create an immutable input, or `new_mut` to create a modifiable one.
/// See also [TensorBuilder](struct.TensorBuilder.html).
#[derive(Clone, Debug)]
pub(crate) struct Input {
    pub(crate) id: usize,
    pub(crate) mut_usage: bool,
}

impl<'graph> Input {
    /// Instantiates a new immutable `Input` object.
    ///
    /// Run-time value of `val` is passed as an `ndarray::ArrayView` or `ndarray::ArrayViewMut`
    /// in `Op::compute` depending on `mut_usage`.
    #[inline]
    pub(crate) fn new<F: Float>(val: &Tensor<'graph, F>, mut_usage: bool) -> Input {
        Input {
            id: val.id(),
            mut_usage,
        }
    }

    #[inline(always)]
    pub(crate) fn as_tensor<F: Float>(&self, graph: &'graph GraphRepr<F>) -> Tensor<'graph, F> {
        graph.tensor(self.id)
    }

    #[inline]
    pub(crate) fn variable_id<F: Float>(&self, graph: &GraphRepr<F>) -> Option<VarArrayID> {
        graph.access_inner(self.id).variable_id
    }

    #[inline]
    pub(crate) fn is_placeholder<F: Float>(&self, graph: &GraphRepr<F>) -> bool {
        graph.access_inner(self.id).is_placeholder
    }
}

/// Builder for `ag::Tensor` returned by [Tensor::builder](struct.Tensor.html#method.builder).
///
/// This structure is required only when constructing user-defined `Op`.
/// ```
/// use autograd as ag;
/// use ag::op::{Op, OpError, ComputeContext, GradientContext};
///
/// struct DummyOp {
///    a: f32
/// }
///
/// impl Op<f32> for DummyOp {
///     fn compute(&self, _: &mut ComputeContext<f32>) -> Result<(), OpError> { Ok(()) }
///     fn grad(&self, _: &mut GradientContext<f32>) {}
/// }
///
/// ag::run(|g: &mut ag::Graph<f32>| {
///     let input = &g.zeros(&[0]);
///     let my_output: ag::Tensor<_> = ag::Tensor::builder(g)
///         .append_input(input, false) // immutable input
///         .append_input(input, true)  // mutable input
///         .build(DummyOp {a: 42.});
/// });
/// ```
pub struct TensorBuilder<'g, F: Float> {
    graph: &'g GraphRepr<F>,
    shape: Option<usize>,
    in_nodes: op::InputArray<Input>,
    can_have_gradient: bool,
    is_placeholder: bool,
    input_indices: Option<op::InputArray<usize>>,
    backprop_inputs: Option<op::InputArray<Input>>,
    known_shape: Option<KnownShape>,
    variable_id: Option<VarArrayID>,
}

pub(crate) struct KnownShape {
    shape: Vec<isize>,
    #[allow(dead_code)]
    is_fully_defined: bool,
}

impl KnownShape {
    pub(crate) fn new(shape: Vec<isize>) -> Self {
        let mut is_fully_defined = true;
        for &a in &shape {
            if a == -1 {
                is_fully_defined = false;
            } else if a <= -1 || a == 0 {
                panic!("Given shape ({:?}) contains invalid dim size(s)", &shape);
            }
        }
        Self {
            shape,
            is_fully_defined,
        }
    }

    #[inline]
    pub fn get(&self) -> &[isize] {
        self.shape.as_slice()
    }

    pub fn validate(&self, target: &[usize]) -> bool {
        if self.shape.len() != target.len() {
            return false;
        }
        for (&i, &u) in self.shape.iter().zip(target) {
            if i > 0 && i as usize != u {
                return false;
            }
        }
        true
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_fully_defined(&self) -> bool {
        self.is_fully_defined
    }
}


#[test]
fn test_topo_order() {
    use crate::tensor_ops as T;
    crate::with(|s| {
        let a: Tensor<f32> = s.zeros(&[4, 2]);
        let v: Tensor<f32> = s.zeros(&[2, 3]);
        let b: Tensor<f32> = s.zeros(&[4, 3]);
        let z = T::matmul(a, v) + b;
        let mut vars = [a.inner(), v.inner(), b.inner(), z.inner()];
        // `sort_by_key` don't reverse the order of `a` and `v`
        vars.sort_by_key(|a| a.topo_rank);
        assert_eq!(vars[0].id, a.id);
        assert_eq!(vars[1].id, v.id);
        assert_eq!(vars[2].id, b.id);
        assert_eq!(vars[3].id, z.id);
    });
}

impl<'graph, F: Float> TensorBuilder<'graph, F> {
    #[inline]
    pub(crate) fn set_variable(mut self, s: VarArrayID) -> TensorBuilder<'graph, F> {
        self.variable_id = Some(s);
        self
    }

    #[inline]
    pub(crate) fn set_known_shape(mut self, s: Vec<isize>) -> TensorBuilder<'graph, F> {
        self.known_shape = Some(KnownShape::new(s));
        self
    }

    #[inline]
    pub(crate) fn set_shape(mut self, s: &Tensor<'graph, F>) -> TensorBuilder<'graph, F> {
        self.shape = Some(s.id());
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, differentiable: bool) -> TensorBuilder<'graph, F> {
        self.can_have_gradient = differentiable;
        self
    }

    #[inline]
    /// Appends input tensor.
    ///
    /// `bool` indicates whether this tensor should be treated as mutable input or not.
    pub fn append_input<T: AsRef<Tensor<'graph, F>>>(
        mut self,
        tensor: T,
        mut_usage: bool,
    ) -> TensorBuilder<'graph, F> {
        let t = tensor.as_ref();
        crate::graph::assert_same_graph(t.graph, self.graph);
        self.in_nodes.push(Input::new(t, mut_usage));
        self
    }

    #[inline]
    pub(crate) fn set_is_placeholder(mut self, a: bool) -> TensorBuilder<'graph, F> {
        self.is_placeholder = a;
        self
    }

    #[inline]
    pub(crate) fn set_input_indices(mut self, a: &[usize]) -> TensorBuilder<'graph, F> {
        self.input_indices = Some(op::InputArray::from_slice(a));
        self
    }

    #[inline]
    /// Append the given tensor to the backprop-input-list.
    ///
    /// Not required unless backprop-inputs are differs from normal-case inputs
    pub fn append_backprop_input<T: AsRef<Tensor<'graph, F>>>(
        mut self,
        a: T,
    ) -> TensorBuilder<'graph, F> {
        crate::graph::assert_same_graph(a.as_ref().graph, self.graph);
        if let Some(ref mut inputs) = self.backprop_inputs {
            inputs.push(Input::new(a.as_ref(), false));
        } else {
            let mut inputs = InputArray::new();
            inputs.push(Input::new(a.as_ref(), false));
            self.backprop_inputs = Some(inputs);
        }
        self
    }

    /// Finalizes this builder and creates a tensor with given `Op` in the graph.
    pub fn build<O>(self, op: O) -> Tensor<'graph, F>
    where
        O: op::Op<F> + 'static,
    {
        let graph = self.graph;
        let rank = if self.in_nodes.is_empty() {
            0
        } else {
            self.in_nodes
                .iter()
                .map(|a| graph.access_inner(a.id).topo_rank)
                .max()
                .map(|a| a + 1)
                .unwrap_or(0)
        };

        let input_indices = if let Some(a) = self.input_indices {
            assert_eq!(
                a.len(),
                self.in_nodes.len(),
                "input_indices.len() must match inputs length"
            );
            a
        } else {
            smallvec::smallvec!(0; self.in_nodes.len())
        };

        let new = TensorInternal {
            // `id` is set in `Graph::install`
            id: usize::default(),
            op: Some(Box::new(op)),
            in_edges: self.in_nodes,
            topo_rank: rank,
            shape: self.shape,
            is_placeholder: self.is_placeholder,
            is_differentiable: self.can_have_gradient,
            input_indices,
            backprop_inputs: self.backprop_inputs,
            known_shape: self.known_shape,
            variable_id: self.variable_id,
        };
        Tensor {
            id: graph.install(new),
            graph,
        }
    }
}

pub(crate) struct Dummy;

impl<T: Float> crate::op::Op<T> for Dummy {
    fn compute(&self, _: &mut crate::op::ComputeContext<T>) -> Result<(), OpError> {
        Ok(())
    }
    fn grad(&self, _: &mut GradientContext<T>) {}
}

use crate::tensor_ops as T;

// -- std::tensor_ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<'b, F: Float> $trt<F> for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: F) -> Self::Output {
                T::$func(&self, &self.graph.scalar(rhs))
            }
        }

        // &Tensor op Float
        impl<'l, 'b, F: Float> $trt<F> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: F) -> Self::Output {
                T::$func(self, &self.graph.scalar(rhs))
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<'r, 'b, F: Float> $trt<Tensor<'b, F>> for $scalar_type {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(&rhs.graph.scalar(F::from(self).unwrap()), &rhs)
            }
        }

        // primitive op &Tensor
        impl<'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for $scalar_type {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(&rhs.graph.scalar(F::from(self).unwrap()), rhs)
            }
        }
    };
}

impl_bin_op_between_tensor_and_float_trait!(Add, add, AddOp);
impl_bin_op_between_tensor_and_float_trait!(Sub, sub, SubOp);
impl_bin_op_between_tensor_and_float_trait!(Mul, mul, MulOp);
impl_bin_op_between_tensor_and_float_trait!(Div, div, DivOp);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f64);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f64);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f64);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f64);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f32);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f32);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f32);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f32);

macro_rules! impl_bin_op_between_tensors {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Tensor
        impl<'b, F: Float> $trt for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(&self, &rhs)
            }
        }

        // Tensor op &Tensor
        impl<'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(&self, rhs)
            }
        }

        // &Tensor op Tensor
        impl<'l, 'b, F: Float> $trt<Tensor<'b, F>> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: Tensor<'b, F>) -> Self::Output {
                T::$func(self, &rhs)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'l, 'r, 'b, F: Float> $trt<&'r Tensor<'b, F>> for &'l Tensor<'b, F> {
            type Output = Tensor<'b, F>;
            fn $func(self, rhs: &'r Tensor<'b, F>) -> Self::Output {
                T::$func(self, rhs)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);

/// Implementors can be converted to `Tensor`.
pub trait AsTensor<'graph, F: Float> {
    fn as_tensor(&self, graph: &'graph impl AsGraphRepr<F>) -> Tensor<'graph, F>;
}

impl<'graph, F: Float> AsTensor<'graph, F> for Tensor<'graph, F> {
    fn as_tensor(&self, _: &'graph impl AsGraphRepr<F>) -> Tensor<'graph, F> {
        *self
    }
}

macro_rules! impl_as_tensor_for_array {
    ($num_elems:expr) => {
        impl<'graph, F: Float, I: crate::Int> AsTensor<'graph, F> for [I; $num_elems] {
            fn as_tensor(&self, graph: &'graph impl AsGraphRepr<F>) -> Tensor<'graph, F> {
                let vec = self
                    .iter()
                    .map(|&a| F::from(a).unwrap())
                    .collect::<Vec<F>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                graph.as_graph_repr().convert_to_tensor(arr)
            }
        }
    };
}

impl_as_tensor_for_array!(0);
impl_as_tensor_for_array!(1);
impl_as_tensor_for_array!(2);
// impl_as_tensor_for_array!(3);
impl_as_tensor_for_array!(4);
impl_as_tensor_for_array!(5);
impl_as_tensor_for_array!(6);
impl_as_tensor_for_array!(7);
impl_as_tensor_for_array!(8);

impl<'graph, F: Float, I: crate::Int> AsTensor<'graph, F> for [I; 3] {
    fn as_tensor(&self, graph: &'graph impl AsGraphRepr<F>) -> Tensor<'graph, F> {
        let vec = self
            .iter()
            .map(|&a| F::from(a).unwrap())
            .collect::<Vec<F>>();

        let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
        graph.as_graph_repr().convert_to_tensor(arr)
    }
}
