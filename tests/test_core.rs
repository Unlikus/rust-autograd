extern crate autograd as ag;
extern crate ndarray;

use ag::tensor_ops as T;
use ag::NdArray;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &'static str {
        "MultiOutputOp"
    }

    fn compute(&self, ctx: &mut ag::op::ComputeContext<f32>) -> Result<(), ag::op::OpError> {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        ctx.append_output(a);
        ctx.append_output(b);
        Ok(())
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f32>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}

#[test]
fn test_nth_tensor() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let a = ag::Tensor::builder(g).build(MultiOutputOp);
        let b = T::nth_tensor(a, 1);
        let c = T::exp(b);
        g.eval(&[c], &[]);
    });
}

#[test]
fn test_hook() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let a: ag::Tensor<f64> = g.ones(&[4, 2]).show();
        let b: ag::Tensor<f64> = g.zeros(&[2, 3]).show_shape();
        let c = T::matmul(a, b).print("aaa");
        g.eval(&[c], &[]);
    });
    ctx.run(|g| {
        let x = g.placeholder(&[]);
        let y = g.placeholder(&[]);
        let z = 2. * x * x + 3. * y + 1.;

        // dz/dy
        let gy = &T::grad(&[z], &[y])[0];
        println!("{:?}", gy.eval(&[], g)); // => Some(3.)

        // dz/dx (requires to fill the placeholder `x`)
        let gx = &T::grad(&[z], &[x])[0];
        println!("{:?}", gx.eval(&[x.given(ag::ndarray::arr0(2.).view())], g)); // => Some(8.)

        // ddz/dx (differentiates `z` again)
        let ggx = &T::grad(&[gx], &[x])[0];
        println!("{:?}", ggx.eval(&[], g)); // => Some(4.)
    });
}

#[test]
fn test_many_nodes() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        for _ in 0..10000 {
            let x = g.placeholder(&[3]);
            let z = 2.0 * x / 2.0 / 2.0;
            T::grad(&[z], &[x])[0];
        }
    });
}

#[test]
fn owned_and_borrowed_array_at_runtime() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x: ag::Tensor<f32> = g.ones(&[6]);
        let y: ag::Tensor<f32> = g.ones(&[6]);
        let mut k = x + y;
        let z = T::reshape(x, &[2, 3]);
        for _ in 0..10000 {
            k = k + y;
        }
        let re = g.eval(&[x, k, z], &[]);
        println!("{:?}", re);
    });
}
