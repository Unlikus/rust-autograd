//! Defining things related to `ag::Graph`.


use crate::variable::{VarArrayID, FullName, NamespaceTrait, VariableNamespace};
use crate::tensor::{Tensor, TensorInternal};
use crate::{Float, FxHashMap, VariableEnvironment};
use smallvec::alloc::borrow::Cow;
use std::fmt;
use std::ops::Deref;
use std::{cell::RefCell, collections::HashMap};
use std::cell::{Ref, RefMut};
use crate::op::Op;

type TensorID = usize;

pub struct GraphRepr<F: Float> {
    pub(crate) node_set: RefCell<Vec<TensorInternal<F>>>,
    pub(crate) variable2node: RefCell<FxHashMap<VarArrayID, TensorID>>,
}

impl<'t, 'g, F: Float> GraphRepr<F> {
    #[inline]
    pub(crate) fn install(&'g self, mut node: TensorInternal<F>) -> TensorID {
        let mut inner = self.node_set.borrow_mut();
        let id = inner.len();
        node.id = id;
        inner.push(node);
        id
    }

    #[inline(always)]
    // `i` must be an id returned by Graph::install
    pub(crate) fn access_inner(&self, i: TensorID) -> Ref<TensorInternal<F>> {
        let borrow = self.node_set.borrow();
        Ref::map(borrow, |t| &t[i])
    }

    // `i` must be an id returned by Graph::install
    #[inline(always)]
    pub(crate) fn access_inner_mut(&self, i: TensorID) -> RefMut<TensorInternal<F>> {
        let borrow = self.node_set.borrow_mut();
        RefMut::map(borrow, |t| &mut t[i])
    }

    #[inline(always)]
    pub(crate) fn tensor(&'g self, id: TensorID) -> Tensor<'g, F> {
        Tensor { id, graph: self }
    }

    #[inline]
    pub fn variable_by_id(&'g self, vid: VarArrayID) -> Tensor<'g, F> {
        let tid = {
            let temp = self.variable2node.borrow();
            temp.get(&vid).cloned()
        };
        if let Some(tid) = tid {
            // use existing tensor
            self.tensor(tid)
        } else {
            // allocate new tensor
            let allocated = Tensor::builder(self)
                .set_variable(vid)
                .build(crate::tensor_ops::basic_source_ops::Variable);
            // register vid -> tid map
            self.variable2node.borrow_mut().insert(vid, allocated.id);
            allocated
        }
    }

    #[inline]
    pub fn variable_by_name<S: AsRef<str>>(
        &self,
        name: S,
        namespace: &impl NamespaceTrait<F>,
    ) -> Tensor<F> {
        let full_name = &FullName::new(namespace.name(), Cow::Borrowed(name.as_ref()));
        if let Some(&vid) = namespace.env().name_to_id.get(full_name) {
            // find VariableID
            self.variable_by_id(vid)
        } else {
            let ns = namespace.name();
            if ns == "" {
                panic!(
                    "variable array not found in default namespace: {}",
                    name.as_ref()
                )
            } else {
                panic!("variable array not found in `{}`: {}", ns, name.as_ref())
            }
        }
    }

    /// Returns a map of `Tensor`s associated with the ids.
    ///
    /// See `VariableEnvironment` for the usages.
    pub fn var_tensors_by_id<'e: 'g>(
        &'g self,
        env: &'e VariableEnvironment<F>,
    ) -> impl Iterator<Item=(VarArrayID, Tensor<'g, F>)> {
        (0..env.array_list.len())
            .map(move |vid| (vid.into(), self.variable_by_id(vid.into())))
    }

    /// Returns a map of `Tensor`s associated with the names in the specified namespace.
    ///
    /// See `VariableEnvironment` for the usages.
    pub fn var_tensors_by_name<'e: 'name + 'g, 'name>(
        &'g self,
        ns: &'name VariableNamespace<'e, 'name, F>,
    ) -> impl Iterator<Item=(&'name str, Tensor<'g, F>)> {
        ns.env()
            .name_to_id
            .iter()
            .filter_map(move |ent| {
                // filter out other namespaces
                if &ent.0.namespace_id == ns.name() {
                    Some((ent.0.variable_name.deref(), self.variable_by_id(*ent.1)))
                } else {
                    None
                }
            })
    }
}

impl<T: Float> fmt::Debug for GraphRepr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let set = &*self.node_set.borrow();
        let mut buf = format!("graph size: {}\n", set.len());
        for node in set {
            buf += format!("{}\n", node).as_str();
        }
        write!(f, "{}", buf)
    }
}

/// Creates and runs a computation graph.
///
/// See also [Graph](struct.Graph.html).
/// ```
/// use autograd as ag;
/// use ag::ndarray;
/// use ag::tensor_ops as T;
///
/// let grad = ag::run(|graph| {
///     let x = graph.placeholder(&[]);
///     let y = graph.placeholder(&[]);
///     let z = 2.*x*x + 3.*y + 1.;
///
///     // dz/dx (symbolic):
///     let grad = &T::grad(&[z], &[x])[0];
///
///     // Evaluate dz/dx when x=3:
///     grad.eval(&[x.given(ndarray::arr0(3.0).view())], graph).unwrap()
/// });
/// assert_eq!(grad, ndarray::arr0(12.0).into_dyn());
/// ```
pub fn run<F, FN, R>(f: FN) -> R
where
    F: Float,
    FN: FnOnce(&mut Graph<F>) -> R,
{
    let env_handle = &mut VariableEnvironment::new();
    let graph_internal = GraphRepr {
        node_set: RefCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(FxHashMap::default()),
    };
    let mut g = Graph {
        env_handle,
        inner: graph_internal,
    };
    f(&mut g)
}

/// Creates a scope for a computation graph.
///
/// Prefer to use [`run`] instead, as that is more flexible.
/// This function is kept for backwards compatibility.
pub fn with<F, FN>(f: FN)
where
    F: Float,
    FN: FnOnce(&mut Graph<F>),
{
    run(f);
}

/// Generator of `Tensor` objects.
///
/// Use [run] or [VariableEnvironment::run] to instantiate this.
pub struct Graph<'env, 'name, F: Float> {
    pub(crate) env_handle: &'env mut VariableEnvironment<'name, F>,
    pub(crate) inner: GraphRepr<F>,
}

impl<'env, 'name, F: Float> Graph<'env, 'name, F> {
    /// Returns a reference of the current VariableEnvironment
    #[inline]
    pub fn env(&self) -> &VariableEnvironment<F> {
        self.env_handle
    }

    /// Returns a mutable reference of the current VariableEnvironment
    #[inline]
    pub fn env_mut(&mut self) -> &mut VariableEnvironment<'name, F> {
        self.env_handle
    }

    /// Get or create a Tensor associated with the given `VarArrayID`.
    #[inline]
    pub fn variable_by_id(&self, vid: VarArrayID) -> Tensor<F> {
        self.inner.variable_by_id(vid)
    }

    #[inline]
    pub(crate) fn inner(&self) -> &GraphRepr<F> {
        &self.inner
    }
}

impl<'env, 'name, F: Float> Deref for Graph<'env, 'name, F> {
    type Target = GraphRepr<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub trait AsGraphRepr<F: Float> {
    fn as_graph_repr(&self) -> &GraphRepr<F>;
}

impl<F: Float> AsGraphRepr<F> for GraphRepr<F> {
    #[inline]
    fn as_graph_repr(&self) -> &GraphRepr<F> {
        self
    }
}

impl<F: Float> AsGraphRepr<F> for Graph<'_, '_, F> {
    #[inline]
    fn as_graph_repr(&self) -> &GraphRepr<F> {
        &self.inner
    }
}

#[inline]
pub(crate) fn assert_same_graph<F: Float>(a: &impl AsGraphRepr<F>, b: &impl AsGraphRepr<F>) {
    assert_eq!(
        a.as_graph_repr() as *const _,
        b.as_graph_repr() as *const _,
        "Detected tensors belonging to different graphs"
    );
}

#[test]
#[should_panic]
fn test_mixed_graph() {
    crate::VariableEnvironment::<f32>::new().run(|g| {
        let a = g.zeros(&[1]);
        crate::VariableEnvironment::<f32>::new().run(|g2| {
            let b = g2.zeros(&[1]);
            let _ = a + b;
        });
    });
}

