use std::fmt;
use rand::Rng;

pub struct Tensor {
    pub dim: Vec<usize>,
    pub elems: Vec<f64>,
}

impl Tensor {
    pub fn new(elems: &[f64], dim: &[usize]) -> Tensor {
        let dim_product: usize = dim.iter().product::<usize>();
        if dim_product != elems.len() {
            panic!("Tensor dimensions do not fit number of elements")
        }
        Tensor{ 
            elems: elems.to_vec(),
            dim: dim.to_vec() 
        }
    }

    pub fn n_elems(self: &Tensor) -> usize {
        return self.dim.iter().product::<usize>()
    }

    pub fn reshape(&mut self, dim:&[usize]) {
        let dim_product: usize = dim.iter().product::<usize>();
        if self.n_elems() == dim_product {
            self.dim = dim.to_vec();
        }
        else {
            panic!("{:?} Tensor cannot be reshaped to {:?}", self.dim, dim);
        }
    }

    pub fn T(&mut self) -> Tensor {
        let mut elems = Vec::<f64>::with_capacity(self.elems.len());
        for j in 0..self.dim[1] {
            for i in 0..self.dim[0] {
                elems.push(self.elems[j+(i*self.dim[1])])
            }
        }
        Tensor{ elems: elems, dim: vec![self.dim[1], self.dim[0]] }
    }

    pub fn linspace(start: f64, end: f64, num: usize) -> Tensor {
        let mut elems = Vec::<f64>::with_capacity(num);
        let stepsize: f64 = (&end - &start) / ((num as f64) - 1.0);
        for i in 0..num {
            elems.push(start + (i as f64)*stepsize)
        }
        Tensor { elems: elems, dim: vec![1, num] }
    }

    pub fn rand(min: f64, max: f64, num: usize) -> Tensor {
        let range: f64 = max - min;
        let mut elems = Vec::<f64>::with_capacity(num);
        let mut rng = rand::thread_rng();
        for _i in 0..num {
            let r: f64 = rng.gen();
            elems.push((r * range) - min);
        }
        Tensor { elems: elems, dim: vec![1, num] }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let step: usize = self.dim[1];
        for i in 0..self.dim[0] {
            let start: usize = i*self.dim[1];

            //rounds to 3 decimal places (10^3)
            let slice = &self.elems[start..(start+step)]
                .iter()
                .map(|x| ((x*1000.).round() / 1000.).to_string()) 
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, " [{}]\n", slice.to_owned());
        }
        Ok(())
    }
}
