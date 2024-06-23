use tensors::Tensor;
pub mod tensors;
pub mod ops;
use std::time::Instant;


fn main() {
    let mut t4 = Tensor::linspace(0., 16777215., 16777216);
    t4.reshape(&[4096, 4096]);
    let mut t5 = Tensor::linspace(0., 16777215., 16777216);
    t5.reshape(&[4096, 4096]);
    let now = Instant::now();
    {
    let t6 = t4*t5;
    }
    let elapsed = now.elapsed();
    println!("{:?}", elapsed);
}

