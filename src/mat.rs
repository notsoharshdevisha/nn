use crate::float::Float;

struct Mat<T: Float> {
    val: Vec<Vec<T>>,
}

impl<T: Float> Mat<T> {
    fn check_dimensions(data: &[Vec<T>]) {
        if data.is_empty() || data[0].is_empty() {
            panic!("Cannot Initialize Empty Matrix");
        }
    }

    pub fn new(data: Vec<Vec<T>>) -> Self {
        Self::check_dimensions(&data);
        Mat { val: data }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.val.len(), self.val[0].len())
    }

    pub fn dot(&self, other: &Self) -> Self {
        let (m1, n1) = self.shape();
        let (m2, n2) = other.shape();
        assert_eq!(n1, m2, "Dimension Mismatch for dot product");

        let mut res = vec![vec![T::default(); n2]; m1];

        for i in 0..m1 {
            for j in 0..n2 {
                for k in 0..n1 {
                    res[i][j] += self.val[i][k] * other.val[k][j];
                }
            }
        }
        Mat { val: res }
    }
}

mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_empty_matrix() {
        let _: Mat<f64> = Mat::new(vec![]);
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
        assert_eq!(result.val, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }
}
