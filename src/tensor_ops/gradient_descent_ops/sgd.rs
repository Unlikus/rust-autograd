use crate::Float;

pub(crate) struct SGDOp<T: Float> {
    pub lr: T,
}

impl<T: Float> SGDOp<T> {
    pub(crate) fn new(lr: T) -> Self {
        SGDOp { lr }
    }
}

impl<T: Float> crate::op::Op<T> for SGDOp<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let coef = -self.lr;
        let updates = ctx.input(1).map(|x| x.clone() * coef);
        ctx.input_mut(0)
            .zip_mut_with(&updates, |l, r| *l += r.clone());
        ctx.append_output(updates);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}
