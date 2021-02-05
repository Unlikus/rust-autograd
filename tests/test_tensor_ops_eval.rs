extern crate autograd as ag;
extern crate ndarray;
use self::ag::NdArray;
use ag::tensor_ops as T;
use ag::VariableEnvironment;
use ndarray::array;

#[test]
fn reduce_prod() {
    let mut env = VariableEnvironment::new();
    env.run(|g| {
        let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
        let v = g.convert_to_tensor(rng.standard_normal(&[3, 2]));
        let z = T::reduce_prod(v, &[0, 1], false); // keep_dims=false
        let empty_shape: &[usize] = &[];
        assert_eq!(z.eval(&[], g).unwrap().shape(), empty_shape);
    });
}

#[test]
fn argmax() {
    let mut env = VariableEnvironment::new();
    env.run(|g| {
        let x = g.convert_to_tensor(array![[3., 4.], [5., 6.]]);
        let y = T::argmax(x, -1, false);
        assert_eq!(y.eval(&[], g), Ok(ndarray::arr1(&[1., 1.]).into_dyn()));
    });
}

#[test]
fn argmax_with_multi_max_args() {
    let mut env = VariableEnvironment::new();
    env.run(|g| {
        let x = g.convert_to_tensor(array![1., 2., 3., 3.]);
        let y = T::argmax(x, 0, false);
        assert_eq!(2., y.eval(&[], g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_mean() {
    let mut env = VariableEnvironment::new();
    let v = env.slot().set(array![2., 3., 4.]);
    env.run(|g| {
        let v = g.variable_by_id(v);
        let z = T::reduce_mean(v, &[0], false); // keep_dims=false
        assert_eq!(3., z.eval(&[], g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_grad() {
    let mut env = VariableEnvironment::new();
    let v = env.slot().set(array![2., 3., 4.]);
    env.run(|g| {
        let v = g.variable_by_id(v);
        let z = T::reduce_mean(v, &[0], false); // keep_dims=false
        let grad = T::grad(&[z], &[v])[0];
        assert_eq!(grad.eval(&[], g).unwrap().shape(), &[3]);
    });
}

#[test]
fn transpose_matmul_square() {
    let mut env = VariableEnvironment::new();
    env.run(|g| {
        let x = g.convert_to_tensor(array![[0., 1.], [2., 3.]]);
        let w = g.convert_to_tensor(array![[0., 1.], [2., 3.]]);
        let w2 = T::transpose(w, &[1, 0]);
        let mm = T::matmul(x, w2);
        assert_eq!(
            mm.eval(&[], g).unwrap().as_slice().unwrap(),
            &[1., 3., 3., 13.]
        );
    });
}

#[test]
fn transpose_matmul() {
    let mut env = VariableEnvironment::new();
    env.run(|g| {
        let x = g.convert_to_tensor(array![[0., 1., 2.], [3., 4., 5.]]);
        let w = g.convert_to_tensor(array![[0., 1.], [2., 3.]]);
        let x2 = T::transpose(x, &[1, 0]).show();
        let mm = T::matmul(x2, w).show();
        assert_eq!(
            mm.eval(&[], g).unwrap().as_slice().unwrap(),
            &[6., 9., 8., 13., 10., 17.]
        );
    });
}

#[test]
fn test_mm() {
    let mut env = VariableEnvironment::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a = g.ones(&[2, 5]);
        let b = g.ones(&[5, 1]);
        let c = T::matmul(&a, &b);
        let d = c.eval(&[], g).unwrap();
        assert_eq!(d.as_slice().unwrap(), &[5., 5.]);
    });
}

#[test]
fn test_batch_matmul_normal() {
    // blas is used
    let mut env = VariableEnvironment::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a: ag::Tensor<f32> = g.ones(&[2, 3, 4, 2]);
        let b: ag::Tensor<f32> = g.ones(&[2, 3, 2, 3]);
        let c = T::batch_matmul(a, b);
        let shape = &[2, 3, 4, 3];
        let size = shape.iter().product();
        let ans = NdArray::<_>::from_shape_vec(ndarray::IxDyn(shape), vec![2f32; size]).unwrap();
        let ret = c.eval(&[], g).unwrap();
        ret.all_close(&ans, 1e-4);
    });
}

#[test]
fn test_batch_matmul_trans_not_square() {
    // blas is not used
    let mut env = VariableEnvironment::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let b: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = T::batch_matmul(a, b);
        let ans = array![[[7., 10.], [15., 22.]], [[7., 10.], [15., 22.]]].into_dyn();
        let ret = c.eval(&[], g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_both() {
    // blas is not used
    let mut env = VariableEnvironment::<f32>::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let b_: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> = T::transpose(b_, &[0, 2, 1]);
        let c = T::batch_matmul(a, b);
        let ans = array![[[7., 15.], [10., 22.]], [[7., 15.], [10., 22.]]].into_dyn();
        let ret = c.eval(&[], g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_lhs() {
    // blas is used
    let mut env = VariableEnvironment::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = T::batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(&[], g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_with_copy() {
    // blas is used
    let mut env = VariableEnvironment::new();
    env.run(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> =
            g.convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = T::batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(&[], g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}
