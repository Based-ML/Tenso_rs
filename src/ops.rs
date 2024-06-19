use std::ops::{Add, Sub, Mul};
use crate::tensors::Tensor;


impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        if self.dim != rhs.dim {
            panic!("Tensors must be of same dimension for addition.")
        }
        let n_elems: usize = self.n_elems();
        let mut new_elems = Vec::<f64>::with_capacity(n_elems);
        for i in 0..n_elems {
            new_elems.push(self.elems[i] + rhs.elems[i])
        }
        Tensor{
            elems: new_elems,
            dim: self.dim,
        }
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Tensor {
        let n_elems: usize = self.n_elems();
        let mut new_elems = Vec::<f64>::with_capacity(n_elems);
        for i in 0..n_elems {
            new_elems.push(self.elems[i] + (rhs as f64))
        }
        Tensor{
            elems: new_elems,
            dim: self.dim,
        }
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        if self.dim != rhs.dim {
            panic!("Tensors must be of same dimension for addition.")
        }
        let n_elems: usize = self.n_elems();
        let mut new_elems = Vec::<f64>::with_capacity(n_elems);
        for i in 0..n_elems {
            new_elems.push(self.elems[i] - rhs.elems[i])
        }
        Tensor{
            elems: new_elems,
            dim: self.dim,
        }
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Tensor {
        let n_elems: usize = self.n_elems();
        let mut new_elems = Vec::<f64>::with_capacity(n_elems);
        for i in 0..n_elems {
            new_elems.push(self.elems[i] - (rhs as f64))
        }
        Tensor{
            elems: new_elems,
            dim: self.dim,
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        let n_elems: usize = self.n_elems();
        let mut new_elems = Vec::<f64>::with_capacity(n_elems);
        for i in 0..n_elems {
            new_elems.push(self.elems[i] * (rhs as f64))
        }
        Tensor{
            elems: new_elems,
            dim: self.dim,
        }
    }
}

impl Mul<Tensor>  for Tensor {
    type Output = Tensor;

    fn mul(self, mut rhs: Tensor) -> Tensor {
        if self.dim[1] != rhs.dim[0] {
            panic!("Tensors not compatible for Multiplication")
        }
        let dim = vec![self.dim[0], rhs.dim[1]];
        let mut elems = Vec::<f64>::with_capacity(dim.iter().product());
        rhs = rhs.T(); // transposing alligns vector rows and columns
        for i in 0..self.dim[0] {
            for j in 0..self.dim[0] {
                let mut e: f64 = 0.;
                for k in 0..self.dim[1] {
                    // println!("self index = {}",(i*self.dim[1])+k);
                    // println!("rhs.T index = {}",(i*self.dim[1])+k);
                    e += self.elems[(i*self.dim[1])+k] * rhs.elems[(j*rhs.dim[1])+k];
                }
                elems.push(e);
            }
        }
        Tensor{ elems: elems, dim: vec![self.dim[0], rhs.dim[0]] }
    }
}
