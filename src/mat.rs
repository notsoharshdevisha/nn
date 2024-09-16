use std::{
    cmp::Eq,
    ops::{Add, Mul},
};

use crate::float::Float;

#[derive(Debug, PartialEq, Eq)]
struct Mat<T: Float> {
    rows: Vec<Vec<T>>,
}

impl<T: Float> Mat<T> {
    fn check_dimensions(data: &[Vec<T>]) {
        if data.is_empty() || data[0].is_empty() {
            panic!("Cannot Initialize Empty Matrix");
        }

        let n = data[0].len();
        for row in data.iter() {
            if row.len() != n {
                panic!("Inconsistent Row Lengths");
            }
        }
    }

    pub fn new(data: Vec<Vec<T>>) -> Self {
        Self::check_dimensions(&data);
        Mat { rows: data }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.rows[0].len())
    }

    pub fn val_at(&self, i: usize, j: usize) -> T {
        self.rows[i][j]
    }

    pub fn dot(&self, other: &Mat<T>) -> Self {
        let (m1, n1) = self.shape();
        let (m2, n2) = other.shape();
        assert_eq!(n1, m2, "Dimension Mismatch for dot product");

        let mut res = vec![vec![T::default(); n2]; m1];

        for i in 0..m1 {
            for j in 0..n2 {
                for k in 0..n1 {
                    res[i][j] += self.val_at(i, k) * other.val_at(k, j);
                }
            }
        }
        Mat { rows: res }
    }
}

impl<T: Float> Add<&Mat<T>> for &Mat<T> {
    type Output = Mat<T>;
    fn add(self, other: &Mat<T>) -> Mat<T> {
        let (m1, n1) = self.shape();
        let (m2, n2) = other.shape();
        assert_eq!(m1, m2, "Dimension Mismatch for addition");
        assert_eq!(n1, n2, "Dimension Mismatch for addition");
        let mut res = vec![vec![T::default(); n1]; m1];
        for i in 0..m1 {
            for j in 0..n1 {
                res[i][j] = self.rows[i][j] + other.rows[i][j];
            }
        }
        Mat { rows: res }
    }
}

impl<T: Float> Mul<&Mat<T>> for &Mat<T> {
    type Output = Mat<T>;
    fn mul(self, other: &Mat<T>) -> Mat<T> {
        self.dot(&other)
    }
}

mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_empty_matrix() {
        let _: Mat<f32> = Mat::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_empty_matrix2() {
        let _: Mat<f64> = Mat::new(vec![vec![]]);
    }

    #[test]
    fn test_shape() {
        let m = Mat::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(m.shape(), (2, 3));
    }

    #[test]
    fn test_dot_product() {
        let v1: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v2: Vec<Vec<f32>> = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let m1 = Mat::new(v1);
        let m2 = Mat::new(v2);
        let result = m1.dot(&m2);
        assert_eq!(result.rows, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }

    #[test]
    fn test_add_matrices_same_dimensions() {
        let mat1 = Mat::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let mat2 = Mat::new(vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
        let expected_result = Mat::new(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);

        let result = &mat1 + &mat2;

        for i in 0..result.rows.len() {
            for j in 0..result.rows[0].len() {
                assert_eq!(result.val_at(i, j), expected_result.val_at(i, j));
            }
        }
    }

    #[test]
    #[should_panic(expected = "Dimension Mismatch for addition")]
    fn test_add_matrices_different_dimensions() {
        let mat1 = Mat::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let mat2 = Mat::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let _result = &mat1 + &mat2;
    }

    #[test]
    fn test_mul_matrices() {
        let mat1 = Mat::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let mat2 = Mat::new(vec![vec![2.0, 0.0], vec![1.0, 2.0]]);
        let result = &mat1 * &mat2;
        let expected = Mat::new(vec![vec![4.0, 4.0], vec![10.0, 8.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_identity() {
        let mat = Mat::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let identity = Mat::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let result = &mat * &identity;
        assert_eq!(result, mat);
    }

    #[test]
    fn test_mul_zero_matrix() {
        let mat = Mat::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let zero_matrix = Mat::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let result = &mat * &zero_matrix;
        let expected = Mat::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_mul_incompatible_matrices() {
        let mat1 = Mat::new(vec![vec![1.0, 2.0]]);
        let mat2 = Mat::new(vec![vec![1.0, 2.0]]);
        let _result = &mat1 * &mat2;
    }
}
