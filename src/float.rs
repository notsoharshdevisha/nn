use std::ops::{Add, AddAssign, Mul};

/// Trait to represent floating-point types `f32` and `f64`
pub trait Float: Copy + Add<Output = Self> + AddAssign + Mul<Output = Self> + Default {}

impl Float for f32 {}

impl Float for f64 {}
