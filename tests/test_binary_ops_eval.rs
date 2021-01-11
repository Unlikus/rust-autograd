extern crate autograd as ag;
extern crate ndarray;

#[test]
fn scalar_add() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let z: ag::Tensor<f64> = 3. + g.ones(&[3]) + 2.;
        assert_eq!(z.eval(&[], g), Ok(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
    });
}

#[test]
fn scalar_sub() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let ref z: ag::Tensor<f64> = 3. - g.ones(&[3]) - 2.;
        assert_eq!(z.eval(&[], g), Ok(ndarray::arr1(&[0., 0., 0.]).into_dyn()));
    });
}

#[test]
fn scalar_mul() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let ref z: ag::Tensor<f64> = 3. * g.ones(&[3]) * 2.;
        assert_eq!(z.eval(&[], g), Ok(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
    });
}

#[test]
fn scalar_div() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let z: ag::Tensor<f64> = 3. / g.ones(&[3]) / 2.;
        assert_eq!(
            z.eval(&[], g),
            Ok(ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn())
        );
    });
}

#[test]
fn slice() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let ref a: ag::Tensor<f32> = g.zeros(&[4, 4]);
        let ref b = g.slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
        assert_eq!(b.eval(&[], g).unwrap().shape(), &[4, 2]);
    });
}

#[test]
fn slice_negative() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let ref a: ag::Tensor<f32> = g.zeros(&[4, 4]);
        let ref b = g.slice(a, &[0, 0], &[-2, 2]); // numpy equivalent is a[:-1, :2]
        assert_eq!(b.eval(&[], g).unwrap().shape(), &[3, 2]);

        let ref b = g.slice(a, &[0, 0], &[-3, 2]); // numpy equivalent is a[:-1, :2]
        assert_eq!(b.eval(&[], g).unwrap().shape(), &[2, 2]);
    });
}
