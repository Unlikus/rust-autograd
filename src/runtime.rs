use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{self, ComputeContext, InputArray, OpInput};
use crate::smallvec::SmallVec;
use crate::tensor::{Tensor, TensorInternal};
use crate::variable::VarArrayID;
use crate::{Float, GraphRepr};
use crate::{FxHashMap, Graph, VariableEnvironment};
use std::cell::{Ref, RefMut, UnsafeCell};

const NUM_MAX_EVAL_BUF: usize = 8;

type EvalBuf<T> = SmallVec<[T; NUM_MAX_EVAL_BUF]>;

/// Helper structure for batched evaluation.
///
/// `Eval` structure can buffer evaluation targets with useful `push` and `extend` functions
/// and runs batched evaluation.
/// Use this in case [Tensor::eval](tensor/struct.Tensor.html#method.eval)
/// or [Graph::eval](struct.Graph.html#method.eval) doesn't help.
///
/// ```
/// use autograd as ag;
/// use ndarray;
///
/// ag::with(|g| {
///    let a = g.placeholder(&[]);
///    let x = a + a;
///    let y = a * a;
///    let z = a / a;
///
///    ag::Eval::new(g)
///        .push(&x)
///        .extend(&[y, z])
///        .feed(&[a.given(ndarray::arr0(2.).view())])
///        .run();  // Do eval
///    });
/// ```
pub struct Eval<'view, 'feed, 'graph, 'e, 'n, 'c, F: Float> {
    scope: &'c Graph<'e, 'n, F>,
    buf: EvalBuf<Tensor<'graph, F>>,
    feeds: Option<&'feed [crate::runtime::Feed<'view, F>]>,
}

impl<'feed, 'tensor, 'view, 'graph, 'e, 'n, 'c, F: Float>
    Eval<'view, 'feed, 'graph, 'e, 'n, 'c, F>
{
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new(scope: &'c Graph<'e, 'n, F>) -> Self {
        Eval {
            feeds: None,
            scope,
            buf: EvalBuf::new(),
        }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.buf.push(*x.as_ref());
        self
    }

    /// `feeds` is a sequence of `(placeholder-tensor, its value)`
    pub fn feed(&mut self, feeds: &'feed [crate::Feed<'view, F>]) -> &mut Self {
        self.feeds = Some(feeds);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'tensor [A]) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.buf.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'tensor self) -> Vec<Result<NdArray<F>, crate::EvalError>> {
        self.scope
            .eval(self.buf.as_slice(), self.feeds.unwrap_or(&[]))
    }
}

/// Links a placeholder tensor and its value at run-time.
///
/// Use `Tensor::given` to instanciate, and
/// ensure that this is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
///
/// let mut env = ag::VariableEnvironment::new();
///
/// env.run(|g| {
///     let x = g.placeholder(&[2]);
///
///     // Fills the placeholder with an ArrayView, then eval.
///     let value = array![1., 1.];
///     let feed: ag::Feed<_> = x.given(value.view());
///     x.eval(&[feed], g);
/// });
/// ```
pub struct Feed<'feed, T: Float> {
    /// The id of the placeholder tensor
    placeholder_id: usize,
    /// A run-time value of the placeholder
    value: NdArrayView<'feed, T>,
}

impl<'feed, F: Float> Feed<'feed, F> {
    #[inline]
    pub(crate) fn new(placeholder_id: usize, value: NdArrayView<'feed, F>) -> Self {
        Feed {
            placeholder_id,
            value,
        }
    }
}
#[derive(Copy, Clone)]
enum ValueType {
    Owned,
    View,
}

#[derive(Copy, Clone)]
struct ValueInfo {
    ty: ValueType,
    // key to lookup output
    key: usize,
}

impl ValueInfo {
    #[inline]
    fn new(ty: ValueType, key: usize) -> Self {
        ValueInfo { ty, key }
    }
}

struct OutputStorage<'view, F: Float> {
    // - storage itself is not shared between threads
    // - items in the storage never gone while evaluation loop (NdArray's relocation is shallow copy).
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
}

struct OutputStorageInner<'view, F: Float> {
    // Each of NdArray is Some right up until eval's ret-val extraction phase.
    // In that phase, each of entry is replaced with None to avoid copying the entire vector.
    value_storage: Vec<Option<NdArray<F>>>,
    view_storage: Vec<NdArrayView<'view, F>>,
}

impl<'view, F: Float> OutputStorage<'view, F> {
    #[inline]
    fn new() -> Self {
        OutputStorage {
            inner: UnsafeCell::new(OutputStorageInner {
                value_storage: Vec::new(),
                view_storage: Vec::new(),
            }),
        }
    }

    #[inline]
    unsafe fn inner(&self) -> &OutputStorageInner<'view, F> {
        &*self.inner.get()
    }

    #[inline]
    unsafe fn inner_mut(&self) -> &mut OutputStorageInner<'view, F> {
        &mut *self.inner.get()
    }

    #[inline]
    fn push_owned(&self, val: NdArray<F>) -> usize {
        unsafe {
            let s = &mut self.inner_mut().value_storage;
            let ret = s.len();
            s.push(Some(val));
            ret
        }
    }

    #[inline]
    fn push_view(&self, view: NdArrayView<'view, F>) -> usize {
        unsafe {
            let s = &mut self.inner_mut().view_storage;
            let ret = s.len();
            s.push(view);
            ret
        }
    }

    #[inline]
    fn get_from_view(&self, i: usize) -> NdArrayView<'view, F> {
        unsafe { self.inner().view_storage[i].clone() }
    }

    #[inline]
    fn get_from_owned(&self, i: usize) -> NdArrayView<F> {
        unsafe { self.inner().value_storage[i].as_ref().unwrap().view() }
    }

    #[inline]
    fn take_from_owned(&self, i: usize) -> NdArray<F> {
        unsafe { self.inner_mut().value_storage[i].take().unwrap() }
    }

    #[inline]
    fn get(&'view self, vi: ValueInfo) -> NdArrayView<'view, F> {
        match vi.ty {
            ValueType::Owned => self.get_from_owned(vi.key),
            ValueType::View => self.get_from_view(vi.key),
        }
    }
}

// search the feed using `in_node_id`
fn retrieve_feed<'feeds, 'feed, F: Float>(
    feeds: &'feeds [Feed<'feed, F>],
    in_node_id: usize,
) -> NdArrayView<'feeds, F> {
    // linear search is tolerable for feeds in most cases.
    for feed in feeds {
        if feed.placeholder_id == in_node_id {
            return feed.value.view();
        }
    }
    panic!("Placeholder unfilled");
}

// Extract output arrays from `results` and stores into `storage`.
fn install_compute_results<'view, F: Float>(
    ys: Result<op::OutputArray<crate::ArrRepr<'view, F>>, op::OpError>,
    storage: &OutputStorage<'view, F>,
) -> Result<op::OutputArray<ValueInfo>, op::OpError> {
    let mut value_info_list = op::OutputArray::new();
    match ys {
        Ok(ys) => {
            debug_assert!(!ys.is_empty(), "Bad op implementation: empty return value");
            for y in ys {
                match y {
                    crate::ArrRepr::Owned(val) => {
                        let key = storage.push_owned(val);
                        value_info_list.push(ValueInfo::new(ValueType::Owned, key));
                    }
                    crate::ArrRepr::View(val) => {
                        let key = storage.push_view(val);
                        value_info_list.push(ValueInfo::new(ValueType::View, key));
                    }
                };
            }
            Ok(value_info_list)
        }
        Err(e) => Err(e),
    }
}

impl<'e, 'n, F: Float> Graph<'e, 'n, F> {
    /// Evaluates given symbolic tensors as a list of `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    /// Unlike [Tensor::eval](tensor/struct.Tensor.html#method.eval), this function
    /// supports batched evaluation.
    ///
    /// See also [Eval](struct.Eval.html).
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// let mut ctx = ag::VariableEnvironment::<f32>::new();
    /// ctx.run(|g| {
    ///     let a = g.zeros(&[2]);
    ///     let b = g.ones(&[2]);
    ///
    ///     // eval two tensors at once.
    ///     let evaluated = g.eval(&[a, b], &[]);
    ///     assert_eq!(evaluated[0], Ok(array![0., 0.].into_dyn()));
    ///     assert_eq!(evaluated[1], Ok(array![1., 1.].into_dyn()));
    /// });
    /// ```
    pub fn eval<'feed, 'tensor, 'scope, A>(
        &'scope self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
    ) -> Vec<Result<NdArray<F>, crate::EvalError>>
    where
        A: AsRef<Tensor<'scope, F>> + Copy,
    {
        self.inner.eval(tensors, feeds, self.env_handle)
    }
}

struct VariableGuardRegister<'v, F: Float> {
    immutable: Vec<Option<UnsafeCell<Ref<'v, NdArray<F>>>>>,
    mutable: Vec<Option<UnsafeCell<RefMut<'v, NdArray<F>>>>>,
}

impl<'v, 'e, F: Float> VariableGuardRegister<'v, F> {
    fn new(max_size: usize) -> Self {
        let mut immutable = Vec::with_capacity(max_size);
        let mut mutable = Vec::with_capacity(max_size);
        // init with None
        for _ in 0..max_size {
            immutable.push(None);
            mutable.push(None);
        }
        Self { immutable, mutable }
    }

    fn set(&mut self, vid: VarArrayID, mut_usage: bool, env: &'v VariableEnvironment<'e, F>) {
        if mut_usage {
            debug_assert!(
                self.mutable[vid.0].is_none(),
                "Bad op impl: taking a variable"
            );
            self.mutable[vid.0] = Some(UnsafeCell::new(env.array_list[vid.0].borrow_mut()));
        } else {
            debug_assert!(self.immutable[vid.0].is_none(), "Bad op impl");
            self.immutable[vid.0] = Some(UnsafeCell::new(env.array_list[vid.0].borrow()));
        }
    }

    fn borrow(&self, vid: VarArrayID, mut_usage: bool) -> OpInput<'v, F> {
        unsafe {
            if mut_usage {
                OpInput::new_mut(
                    (*self.mutable[vid.0]
                        .as_ref()
                        .expect("`set` is not called with this VariableID")
                        .get())
                    .view_mut(),
                )
            } else {
                OpInput::new(
                    (*self.immutable[vid.0]
                        .as_ref()
                        .expect("`set` is not called with this VariableID")
                        .get())
                    .view(),
                )
            }
        }
    }

    fn unset(&mut self, vid: VarArrayID, mut_usage: bool) {
        if mut_usage {
            self.mutable[vid.0] = None;
        } else {
            self.immutable[vid.0] = None;
        }
    }
}

impl<F: Float> GraphRepr<F> {
    fn eval<'feed, 'tensor, 'g, A>(
        &'g self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
        ctx: &VariableEnvironment<F>,
    ) -> Vec<Result<NdArray<F>, crate::EvalError>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
    {
        let mut node_info_map: FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>> =
            FxHashMap::default();

        // Storage in which compute results are stored. Accessed through UnsafeCell.
        let storage = OutputStorage::new();

        let mut variable_guard_register = VariableGuardRegister::new(ctx.array_list.len());

        // Vec<(node_id, is_parent)>
        let mut dfs_stack = Vec::<(usize, bool)>::with_capacity(1 << 10);

        for t in tensors.iter() {
            crate::graph::assert_same_graph(self, t.as_ref().graph);
            dfs_stack.push((t.as_ref().id(), false));
        }

        while let Some((node_id, is_parent)) = dfs_stack.pop() {
            unsafe {
                //  in this block, relocation of Graph::node_set's contents must not be occurred
                let node = self.access_inner(node_id);
                if is_parent {
                    if would_not_visit(&node, &node_info_map) {
                        continue;
                    }

                    // =====================================================================================
                    // Aggregate input values for `node`. if any of the inputs failed, it's a total failure.
                    // =====================================================================================

                    let mut xs = InputArray::new();

                    let mut input_status = Ok(());

                    // Save var guards
                    for (in_node, _) in node.in_edges.iter().zip(&node.input_indices) {
                        if let Some(vid) = in_node.variable_id(self) {
                            // is variable array
                            variable_guard_register.set(vid, in_node.mut_usage, ctx);
                        }
                    }

                    for (in_node, &in_idx) in node.in_edges.iter().zip(&node.input_indices) {
                        // `in_idx` is not 0 only when `in_node` is multi-output op and `node` selects nth value from it using `Graph::nth_tensor`.
                        let x = {
                            if in_node.is_placeholder(self) {
                                Ok(OpInput::new(retrieve_feed(feeds, in_node.id)))
                            } else if let Some(vid) = in_node.variable_id(self) {
                                // is variable array
                                Ok(variable_guard_register.borrow(vid, in_node.mut_usage))
                            } else {
                                // Search the value of input nodes.
                                match &node_info_map.get(&in_node.id).unwrap() {
                                    Err(e) => Err(e.clone()),
                                    Ok(vi_list) => Ok(OpInput::new(storage.get(vi_list[in_idx]))),
                                }
                            }
                        };
                        match x {
                            Ok(x) => xs.push(x),
                            Err(e) => {
                                input_status = Err(e);
                                break;
                            }
                        }
                    }

                    // ====================================================
                    // Run Op::compute() if `node`'s inputs were not failed
                    // ====================================================

                    let installed_node_info = input_status.and_then(|()| {
                        // let mut ctx = ComputeContext::new(node, xs);
                        let mut ctx = ComputeContext::new(xs);
                        let status = node.get_op().compute(&mut ctx);
                        let ret = status.map(|()| ctx.ys);
                        // register compute result
                        let results = install_compute_results(ret, &storage);
                        results
                    });

                    // Release var guards
                    for (in_node, _) in node.in_edges.iter().zip(&node.input_indices) {
                        if let Some(vid) = in_node.variable_id(self) {
                            // is variable array
                            variable_guard_register.unset(vid, in_node.mut_usage);
                        }
                    }

                    // Cache the result
                    node_info_map.insert(node_id, installed_node_info);
                } else {
                    // Update dfs stack
                    dfs_stack.push((node_id, true));
                    // Push children if needed
                    for child in &node.in_edges {
                        let child = self.access_inner(child.id);
                        if !would_not_visit(&child, &node_info_map) {
                            dfs_stack.push((child.id, false));
                        }
                    }
                }
            }
        }

        // Aggregate return values
        let mut ret = Vec::with_capacity(tensors.len());
        for t in tensors {
            let t = t.as_ref();
            let arr = if let Some(vid) = t.get_variable_id() {
                // case 1: variable tensor
                Ok(ctx.array_list[vid.0].clone().into_inner())
            } else if t.is_placeholder() {
                // case 2: placeholder tensor
                Ok(retrieve_feed(feeds, t.id()).to_owned())
            } else {
                // case 3: normal tensor
                match &node_info_map.get(&t.id()).unwrap() {
                    Ok(value_info_list) => match value_info_list[0] {
                        ValueInfo {
                            ty: ValueType::Owned,
                            key,
                        } => Ok(storage.take_from_owned(key)),
                        ValueInfo {
                            ty: ValueType::View,
                            key,
                        } => Ok(storage.get_from_view(key).to_owned()),
                    },
                    Err(e) => {
                        // convert to EvalError
                        Err(crate::EvalError::OpError(e.clone()))
                    }
                }
            };
            ret.push(arr);
        }
        ret
    }
}

use crate::tensor_ops as T;

#[inline]
fn would_not_visit<F: Float>(
    node: &Ref<TensorInternal<F>>,
    info_map: &FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>>,
) -> bool {
    node.is_placeholder || node.is_variable() || info_map.contains_key(&node.id())
}

#[test]
fn test_eval2() {
    let mut ctx = crate::VariableEnvironment::new();
    ctx.run(|g: &mut Graph<f32>| {
        let a = g.ones(&[1, 1]);
        let b = T::sigmoid(a);
        b.eval(&[], g).unwrap();
    })
}

#[test]
fn test_eval() {
    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let v: Tensor<f32> = g.placeholder(&[3, 2, 1]);
        let z = T::reduce_sum(T::squeeze(v, &[2]), &[0, 1], false);
        let grad = T::grad(&[z], &[v]);
        let eval_result = grad[0].eval(&[v.given(crate::ndarray_ext::ones(&[3, 2, 1]).view())], g);
        assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
    })
}

#[test]
fn test_variable_eval() {
    let mut ctx = VariableEnvironment::new();
    let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let arr_clone = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let a = ctx.slot().set(arr);
    ctx.run(|g| {
        let av = g.variable_by_id(a);
        assert_eq!(Ok(arr_clone), av.eval(&[], g));
    });
}

#[test]
fn test_constant_eval() {
    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Ok(arr.clone()), g.convert_to_tensor(arr).eval(&[], g));
    });
}

#[test]
fn test_placeholder_eval() {
    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let arr: NdArray<f32> = crate::ndarray_ext::ones(&[3, 2, 1]);
        let v = g.placeholder(&[3, 2, 1]);
        let eval_result = v.eval(&[v.given(arr.view())], g);
        assert_eq!(Ok(arr), eval_result);
    });
}
