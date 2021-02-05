use crate::graph::Graph;
use crate::{uuid::Uuid, Float, FxHashMap, GraphRepr, NdArray};
use smallvec::alloc::borrow::Cow;
use smallvec::alloc::fmt::Formatter;
use std::cell::{RefCell, UnsafeCell};
use std::ops::Deref;
use std::path::Path;
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::error::Error;
use std::collections::HashMap;

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, Serialize, Deserialize)]
/// Variable array's ID that is unique in a `VariableEnvironment`.
pub struct VarArrayID(pub(crate) usize);

impl From<usize> for VarArrayID {
    fn from(a: usize) -> VarArrayID {
        VarArrayID(a)
    }
}

impl From<VarArrayID> for usize {
    fn from(a: VarArrayID) -> usize {
        a.0
    }
}

impl std::fmt::Display for VarArrayID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub(crate) type Variable<F> = RefCell<NdArray<F>>;

/// Manages variable arrays
///
/// ## Basic usages
/// ```
/// use autograd as ag;
/// use ag::ndarray_ext;
/// use ag::variable::{VarArrayID, NamespaceTrait};
/// use ag::Tensor;
/// use std::collections::HashMap;
///
///
/// let mut env = ag::VariableEnvironment::new();
///
/// // Register variable arrays in the *default* namespace.
/// // `set` method returns the id of given array;
/// let a: VarArrayID = env.slot().set(ndarray_ext::zeros(&[1, 10]));
/// // You can name an array and lookup it later
/// let b: VarArrayID = env.slot().with_name("b").set(ndarray_ext::zeros(&[1, 10]));
///
/// // Register variable arrays in the `my_namespace` namespace.
/// let c: VarArrayID = env.namespace_mut("my_namespace")
///     .slot()
///     .with_name("c")
///     .set(ndarray_ext::zeros(&[1, 10]));
///
/// // Collecting var names in a specific namespace.
/// let names_: Vec<&str> = env.default_namespace().current_var_names();
///
/// // Create and run a graph with the env.
/// for epoch in 0..10 {
///     env.run(|g| {
///         // Preferred way to lookup variable tensors.
///         let var: HashMap<VarArrayID, Tensor<f32>> = g.var_tensors_by_id(g.env()).collect();
///         var[&a];
///         var[&b];
///
///         // Another preferred way to lookup variable tensors, which uses user-defined names.
///         // You must specify a namespace
///         let ns1 = g.env().default_namespace();
///         let ns2 = g.env().namespace("my_namespace");
///         let var: HashMap<&str, Tensor<f32>> = g.var_tensors_by_name(&ns1).collect();
///         var["b"];
///         let var: HashMap<&str, Tensor<f32>> = g.var_tensors_by_name(&ns2).collect();
///         var["c"];
///
///         // Serious way to get `Tensor` representation of a, b and c.
///         g.variable_by_id(a);
///         g.variable_by_name("b", &g.env().default_namespace());
///         g.variable_by_name("c", &g.env().namespace("my_namespace"));
///
///         // ...
///     })
/// }
/// ```
#[derive(Clone)]
pub struct VariableEnvironment<'name, F> {
    pub(crate) array_list: Vec<Variable<F>>,
    pub(crate) name_to_id: FxHashMap<FullName<'name>, VarArrayID>,
}



/// Identifies variable array
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct FullName<'name> {
    pub(crate) namespace_id: Cow<'name, str>,
    pub(crate) variable_name: Cow<'name, str>,
}

/// Anonymous slot where variable array can be registered
///
/// The registered variable array will be kept in the namespace to which this slot associated.
///
/// Please use `VariableNamespaceMut::slot` to instantiate this.
pub struct VariableSlot<'ns, 'env, 'name, F: Float> {
    namespace: &'ns mut VariableNamespaceMut<'env, 'name, F>,
}

/// Named slot where variable array can be registered
///
/// The registered variable array will be kept in the namespace to which this slot associated.
/// You can lookup the array's tensor representation using the name later.
///
/// Please use `VariableSlot::with_name` to instantiate this.
pub struct NamedVariableSlot<'ns, 'env, 'name, F: Float, S: Into<String>> {
    namespace: &'ns mut VariableNamespaceMut<'env, 'name, F>,
    name: S,
}

/// Anonymous slot where variable array can be registered
///
/// The registered variable array will be kept in the *default* namespace.
///
/// Please use `VariableEnvironment::slot` to instantiate this.
pub struct DefaultVariableSlot<'env, 'name, F: Float> {
    env: &'env mut VariableEnvironment<'name, F>,
}

/// Named slot where variable array can be registered
///
/// The registered variable array will be kept in the *default* namespace.
/// You can lookup the array's tensor representation using the name later.
///
/// Please use `DefaultVariableSlot::with_name` to instantiate this.
pub struct NamedDefaultVariableSlot<'env, 'name, F: Float, S: Into<String>> {
    env: &'env mut VariableEnvironment<'name, F>,
    name: S,
}

/// Manages variable arrays using those unique names or IDs.
///
/// Each of the variables managed by autograd is always associated to a single namespace.
/// Please use `VariableNamespaceMut` to register a new variable array.
pub struct VariableNamespace<'env, 'name, F: Float> {
    pub(crate) env: &'env VariableEnvironment<'name, F>,
    pub(crate) namespace_id: &'static str,
}

/// Mutable version of `VariableNamespace`.
///
/// You can register a new variable array with this namespace using `slot` method.
pub struct VariableNamespaceMut<'env, 'name, F: Float> {
    pub(crate) env: &'env mut VariableEnvironment<'name, F>,
    pub(crate) namespace_id: &'static str,
}

impl<'name> FullName<'name> {
    pub(crate) fn new(namespace_id: &'static str, variable_name: Cow<'name, str>) -> Self {
        FullName {
            namespace_id: Cow::Borrowed(namespace_id),
            variable_name,
        }
    }

    pub(crate) fn to_string(&self) -> String {
        let ns = self.namespace_id.deref();
        let name = self.variable_name.deref();
        format!("{}\u{00001}{}", ns, name)
    }
}

pub trait NamespaceTrait<F: Float> {
    /// The name of this namespace
    fn name(&self) -> &'static str;

    /// A reference to the `VariableEnvironment`.
    fn env(&self) -> &VariableEnvironment<F>;

    /// Returns a reference to the variable array with the specified id.
    ///
    /// `VariableID` is returned by the `*Slot::set`.
    #[inline]
    fn get_array_by_id(&self, vid: VarArrayID) -> &RefCell<NdArray<F>> {
        &self.env().array_list[vid.0]
    }

    /// Returns a reference to the variable array with the specified name.
    ///
    /// Returns `None` if the specified name is not valid in this namespace.
    #[inline]
    fn get_array_by_name<S: AsRef<str>>(&self, name: S) -> Option<&RefCell<NdArray<F>>> {
        let name = &FullName::new(self.name(), Cow::Borrowed(name.as_ref()));
        self.env()
            .name_to_id
            .get(name)
            .map(|vid| &self.env().array_list[vid.0])
    }

    /// Lists all the IDs of the variable arrays in this namespace.
    fn current_var_ids(&self) -> Vec<VarArrayID> {
        self.env()
            .name_to_id
            .iter()
            .filter_map(|(v_name, &vid)| {
                if v_name.namespace_id == self.name() {
                    Some(vid)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Lists all the names of the variable arrays in this namespace.
    fn current_var_names(&self) -> Vec<&str> {
        self.env()
            .name_to_id
            .iter()
            .filter_map(|(v_name, _)| {
                if v_name.namespace_id == self.name() {
                    Some(v_name.variable_name.deref())
                } else {
                    None
                }
            })
            .collect()
    }

}

impl<'ns, 'env, 'name, F: Float, S: Into<String>> NamedVariableSlot<'ns, 'env, 'name, F, S> {
    /// Registers the given name and array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VarArrayID {
        register_variable(
            v,
            self.namespace.namespace_id,
            self.name.into(),
            self.namespace.env,
        )
    }
}

impl<'env, 'name, F: Float> DefaultVariableSlot<'env, 'name, F> {
    /// Registers the given array with the *default* namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VarArrayID {
        register_variable(v, "", Uuid::new_v4().to_string(), self.env)
    }

    /// Specifies the name for the array that will be registered.
    pub fn with_name<S: Into<String>>(
        self,
        name: S,
    ) -> NamedDefaultVariableSlot<'env, 'name, F, S> {
        NamedDefaultVariableSlot {
            env: self.env,
            name,
        }
    }
}

impl<'env, 'name, F: Float, S: Into<String>> NamedDefaultVariableSlot<'env, 'name, F, S> {
    /// Registers the given name and array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VarArrayID {
        register_variable(v, "", self.name.into(), self.env)
    }
}

impl<'ns, 'env, 'name, F: Float> VariableSlot<'ns, 'env, 'name, F> {
    /// Registers the given array with the specified namespace.
    pub fn set<D: ndarray::Dimension>(self, v: ndarray::Array<F, D>) -> VarArrayID {
        register_variable(
            v,
            self.namespace.namespace_id,
            Uuid::new_v4().to_string(),
            self.namespace.env,
        )
    }

    /// Specifies the name for the array that will be registered.
    pub fn with_name<S: Into<String>>(self, name: S) -> NamedVariableSlot<'ns, 'env, 'name, F, S> {
        NamedVariableSlot {
            namespace: self.namespace,
            name,
        }
    }
}

fn register_variable<F: Float, D: ndarray::Dimension, S: Into<String>>(
    v: ndarray::Array<F, D>,
    namespace_id: &'static str,
    variable_name: S,
    env: &mut VariableEnvironment<F>,
) -> VarArrayID {
    let vid = FullName::new(namespace_id, Cow::Owned(variable_name.into()));
    let next_id = env.array_list.len().into();
    env.name_to_id.insert(vid, next_id);
    env.array_list.push(RefCell::new(v.into_dyn()));
    next_id
}

impl<'env, 'name, F: Float> NamespaceTrait<F> for VariableNamespace<'env, 'name, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_id
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'env, 'name, F: Float> NamespaceTrait<F> for VariableNamespaceMut<'env, 'name, F> {
    #[inline]
    fn name(&self) -> &'static str {
        self.namespace_id
    }
    #[inline]
    fn env(&self) -> &VariableEnvironment<F> {
        self.env
    }
}

impl<'e: 'name, 'name, F: Float> VariableNamespace<'e, 'name, F> {
    /// Returns an iterator of variable arrays and those names in this namespace
    fn iter(&'name self) -> impl Iterator<Item=(&'name str, &RefCell<NdArray<F>>)> {
        iter(self)
    }
}

impl<'e: 'name, 'name, F: Float> VariableNamespaceMut<'e, 'name, F> {
    /// Returns an iterator of variable arrays and those names in this namespace
    fn iter(&'name self) -> impl Iterator<Item=(&'name str, &RefCell<NdArray<F>>)> {
        iter(self)
    }
}

fn iter<'name, F: Float>(ns: &'name impl NamespaceTrait<F>) -> impl Iterator<Item=(&'name str, &RefCell<NdArray<F>>)> {
    ns.env()
        .name_to_id
        .iter()
        .filter_map(move |ent| {
            // filter out other namespaces
            if &ent.0.namespace_id == ns.name() {
                Some((ent.0.variable_name.deref(), ns.get_array_by_name(ent.0.variable_name.deref()).unwrap()))
            } else {
                None
            }
        })
}
impl<'ns, 'env, 'name, F: Float> VariableNamespaceMut<'env, 'name, F> {
    /// Makes a temporary slot for registering the variable array in this namespace.
    pub fn slot(&'ns mut self) -> VariableSlot<'ns, 'env, 'name, F> {
        VariableSlot { namespace: self }
    }
}

#[derive(Serialize)]
struct SerializableVariableEnvironment<'a, F> {
    array_list: &'a Vec<Variable<F>>,
    name_to_id: FxHashMap<String, VarArrayID>,
}

#[derive(Deserialize)]
struct DeserializedVariableEnvironment<F> {
    array_list: Vec<Variable<F>>,
    name_to_id: FxHashMap<String, VarArrayID>,
}

// f32 save and load
impl<'env, 'name> VariableEnvironment<'name, f32> {
    /// Creates a new VariableEnvironment using the one previously made persistent.
    ///
    /// Returns the result of the execution.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<VariableEnvironment<'name, f32>, Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f32> = Self::deserialize(path)?;
        Self::load_internal(raw)
    }

    /// Initialize this instance using the one previously made persistent.
    pub fn init<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>>{
        let raw: DeserializedVariableEnvironment<f32> = Self::deserialize(path)?;
        let VariableEnvironment { array_list, name_to_id } = Self::load_internal(raw)?;
        self.array_list = array_list;
        self.name_to_id = name_to_id;
        Ok(())
    }
}

// f64 save and load
impl<'env, 'name> VariableEnvironment<'name, f64> {

    /// Creates a new VariableEnvironment using the one previously made persistent.
    ///
    /// Returns the result of the execution.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<VariableEnvironment<'name, f64>, Box<dyn Error>> {
        let raw: DeserializedVariableEnvironment<f64> = Self::deserialize(path)?;
        Self::load_internal(raw)
    }

    /// Initialize this instance using the one previously made persistent.
    pub fn init<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>>{
        let raw: DeserializedVariableEnvironment<f64> = Self::deserialize(path)?;
        let VariableEnvironment { array_list, name_to_id } = Self::load_internal(raw)?;
        self.array_list = array_list;
        self.name_to_id = name_to_id;
        Ok(())
    }
}

impl<'env, 'name, F: Float> VariableEnvironment<'name, F> {
    pub fn new() -> VariableEnvironment<'name, F> {
        Self {
            name_to_id: FxHashMap::default(),
            array_list: Vec::new(),
        }
    }

    /// Returns an iterator of variable arrays and those ids in this env.
    pub fn iter(&'env self) -> impl Iterator<Item=(VarArrayID, &RefCell<NdArray<F>>)> {
        self.array_list.iter().enumerate().map(|(i, v)| {
            (VarArrayID::from(i), v)
        })
    }

    /// Saves the current VariableEnvironment to storage.
    ///
    /// Returns the result of the execution.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let f = File::create(path.as_ref())?;
        serde_json::to_writer(f, &self.prepare_for_serde())?;
        Ok(())
    }

    fn deserialize<T, P: AsRef<Path>>(path: P) -> Result<T, Box<dyn Error>>
        where T: for<'de> Deserialize<'de>
    {
        let f = File::open(path.as_ref())?;
        let ret = serde_json::from_reader(f)?;
        Ok(ret)
    }

    fn load_internal<T>(env: DeserializedVariableEnvironment<T>) -> Result<VariableEnvironment<'name, T>, Box<dyn Error>> {
        let name_to_id: FxHashMap<FullName, VarArrayID> = env.name_to_id.iter().map(|(fullname, &vid)| {
            let mut split = fullname.split("\u{0001}").into_iter();
            let namespace_id = split.next().unwrap().to_owned();
            let var_name = split.next().unwrap().to_owned();
            let fullname = FullName {
                namespace_id: Cow::Owned(namespace_id),
                variable_name: Cow::Owned(var_name)
            };
            (fullname, vid)
        }).collect();

        Ok(VariableEnvironment {
            array_list: env.array_list,
            name_to_id
        })
    }

    fn prepare_for_serde(&self) -> SerializableVariableEnvironment<F> {
        let name_to_id: FxHashMap<String, VarArrayID> = self.name_to_id.iter().map(|(fullname, vid)| {
            (fullname.to_string(), *vid)
        }).collect();
        SerializableVariableEnvironment {
            array_list: &self.array_list,
            name_to_id
        }
    }

    /// Makes a temporary slot for registering the variable array in the *default* namespace.
    pub fn slot(&'env mut self) -> DefaultVariableSlot<'env, 'name, F> {
        DefaultVariableSlot { env: self }
    }

    /// Gets or creates a namespace with specified name.
    ///
    /// The return value is immutable, its usage is limited to variable lookup etc.
    /// Use `namespace_mut` to register variable.
    #[inline]
    pub fn namespace(
        &'env self,
        namespace_id: &'static str,
    ) -> VariableNamespace<'env, 'name, F> {
        VariableNamespace {
            namespace_id,
            env: self,
        }
    }

    /// Gets or creates a mutable namespace with specified name.
    ///
    /// You can register new variable arrays with the namespace.
    #[inline]
    pub fn namespace_mut(
        &'env mut self,
        namespace_id: &'static str,
    ) -> VariableNamespaceMut<'env, 'name, F> {
        VariableNamespaceMut {
            namespace_id,
            env: self,
        }
    }

    /// Gets or creates a *default* namespace.
    ///
    /// The return value is immutable, its usage is limited to variable lookup etc.
    /// Use `namespace_mut` to register variable.
    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, 'name, F> {
        self.namespace("")
    }

    /// Gets or creates a mutable *default* namespace.
    ///
    /// You can register new variable arrays with the default namespace.
    #[inline]
    pub fn default_namespace_mut(&'env mut self) -> VariableNamespaceMut<'env, 'name, F> {
        self.namespace_mut("")
    }

    /// Returns a reference to the variable array with the specified id.
    ///
    /// `VariableID` is returned by the `*Slot::set`.
    #[inline]
    pub fn get_array_by_id(&self, vid: VarArrayID) -> Option<&RefCell<NdArray<F>>> {
        self.array_list.get(vid.0)
    }

    /// Creates a computation graph associated with this `VariableEnvironment`.
    pub fn run<FN, R>(&'env mut self, f: FN) -> R
    where
        FN: FnOnce(&mut Graph<'env, 'name, F>) -> R,
    {
        let g = GraphRepr {
            node_set: RefCell::new(Vec::with_capacity(256)),
            variable2node: RefCell::new(FxHashMap::default()),
        };
        let mut c = Graph {
            env_handle: self,
            inner: g,
        };
        f(&mut c)
    }
}

#[allow(unused)]
fn compile_common_usages() {
    let mut env = VariableEnvironment::<f32>::new();
    let _cur_names_ = env.default_namespace().current_var_names();

    env.run(|g| {
        let ns = g.env().default_namespace();
        let var: HashMap<_, _> = g.var_tensors_by_name(&ns).collect();

        let _v3_ = g.variable_by_name("a", &ns);
        let v = var["a"];
        let v2 = var["a"];
        let ones = g.zeros(&[1]) + v + v2;
        let _ = ones.eval(&[], &g);
    })
}

#[test]
fn save_and_load() {
    use std::fs;
    use crate::approx::AbsDiffEq;

    let dir = "/tmp/autograd/test_save_and_load";
    fs::create_dir_all(dir).unwrap();
    let path = format!("{}/model.json", dir);
    let rng = crate::ndarray_ext::ArrayRng::<f64>::default();

    let mut env = VariableEnvironment::new();
    env.slot().with_name("a").set(rng.standard_normal(&[2, 3]));
    env.slot().with_name("b").set(rng.standard_normal(&[2, 3]));

    // save
    env.save(&path).unwrap();

    // load and assert
    {
        let loaded_env = VariableEnvironment::<f64>::load(&path).unwrap();

        // assert array equalities
        for (vid, array) in env.iter() {
            let loaded_env_map: HashMap<_, _> = loaded_env.iter().collect();
            let loaded_array = loaded_env_map.get(&vid).unwrap();
            assert!(array.abs_diff_eq(*loaded_array, 1e-6));
        }

        assert_eq!(env.name_to_id, loaded_env.name_to_id);
    }
}

#[test]
fn save_and_init() {
    use std::fs;
    use crate::approx::AbsDiffEq;

    let dir = "/tmp/autograd/test_save_and_init";
    fs::create_dir_all(dir).unwrap();
    let path = format!("{}/model.json", dir);
    let rng = crate::ndarray_ext::ArrayRng::<f64>::default();

    let mut env = VariableEnvironment::new();
    let a = env.slot().with_name("a").set(rng.standard_normal(&[2, 3]));
    let b = env.slot().with_name("b").set(rng.standard_normal(&[2, 3]));

    for _ in 0..10 {
        env.run(|g| {
            let a = g.variable_by_id(a);
            let b = g.variable_by_id(b);
            g.env().save(&path).unwrap();
            g.env_mut().init(&path);
        });
    }
}
