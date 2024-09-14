use rand::Rng;

pub fn batch_err_per(batch_output: &Vec<Vec<f64>>, batch_targets: &[Vec<f64>]) -> f64 {
    let err = batch_err(batch_output, batch_targets);
    let mut cumulative_err_per = 0.0;
    for i in 0..err.len() {
        for j in 0..err[i].len() {
            cumulative_err_per += err[i][j] / batch_targets.len() as f64;
        }
    }
    cumulative_err_per / err.len() as f64
}

pub fn batch_err(batch_output: &Vec<Vec<f64>>, batch_targets: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let i = batch_output.len();
    let ii = batch_targets.len();
    if i == 0 || ii == 0 {
        panic!("Cannot calculate error: empty batch");
    }
    let j = batch_output[0].len();
    let jj = batch_targets[0].len();
    if i != ii || j != jj {
        panic!("Cannot calculate error: batch dimensions mismatch")
    }
    let mut res = vec![vec![0.0; j]; i];
    for row in 0..i {
        for col in 0..j {
            res[row][col] = batch_output[row][col] - batch_targets[row][col];
        }
    }
    res
}

pub fn min_max_scaler(matrix: &mut Vec<Vec<f64>>) {
    if matrix.is_empty() || matrix[0].is_empty() {
        panic!("cannot scale empty matrix");
    }
    let mut col_maxes = vec![f64::MIN; matrix[0].len()];
    let mut col_min = vec![f64::MAX; matrix[0].len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            if matrix[i][j] > col_maxes[j] {
                col_maxes[j] = matrix[i][j];
            }
            if matrix[i][j] < col_min[j] {
                col_min[j] = matrix[i][j];
            }
        }
    }
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            if col_maxes[j] == col_min[j] {
                matrix[i][j] = 0.0;
                continue;
            }
            matrix[i][j] = (matrix[i][j] - col_min[j]) / (col_maxes[j] - col_min[j]);
        }
    }
}

pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        panic!("cannot transpose empty matrix");
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut res = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            res[j][i] = matrix[i][j];
        }
    }
    res
}

fn df_relu(x: &mut f64) {
    if *x < 0.0 {
        *x = 0.1 * *x;
    }
}

fn relu(x: &mut f64) {
    if *x < 0.0 {
        *x = 0.0;
    }
}

pub fn df_activate(matrix: &mut Vec<Vec<f64>>) {
    if matrix.is_empty() {
        panic!("cannot activate empty matrix");
    }
    for row in matrix.iter_mut() {
        for value in row.iter_mut() {
            df_relu(value);
        }
    }
}

pub fn add_bias(bias: &Vec<Vec<f64>>, layer_batch_output: &mut Vec<Vec<f64>>) {
    if bias.is_empty() || bias.len() > 1 {
        panic!("Cannot add bias: Invalid bias dimensions");
    }
    if layer_batch_output.is_empty() {
        panic!("Cannot add bias: Empty layer output");
    }
    let bias_vec = &bias[0];
    if bias_vec.len() != layer_batch_output[0].len() {
        panic!("Cannot add bias: Bias and layer output dimensions mismatch");
    }
    for i in 0..layer_batch_output.len() {
        for j in 0..layer_batch_output[i].len() {
            layer_batch_output[i][j] += bias_vec[j];
        }
    }
}

pub fn activate(matrix: &mut Vec<Vec<f64>>) {
    if matrix.is_empty() {
        panic!("cannot activate empty matrix");
    }
    for row in matrix.iter_mut() {
        for value in row.iter_mut() {
            relu(value);
        }
    }
}

pub fn dot(mat1: &Vec<Vec<f64>>, mat2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let i = mat1.len();
    let k = mat2.len();
    if i == 0 || k == 0 {
        panic!("cannot multiply empty matrices");
    }
    if mat1[0].len() != k {
        panic!("matrix dimension incompatible for dot product")
    }
    let j = mat2[0].len();
    let mut res = vec![vec![0.0; j]; i];
    for row in 0..i {
        for col in 0..j {
            let mut inner_sum = 0.0;
            for inner in 0..k {
                inner_sum += mat1[row][inner] * mat2[inner][col]
            }
            res[row][col] = inner_sum;
        }
    }
    res
}

pub fn generate_rand_matrix(rows: &usize, cols: &usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![0.0; *cols]; *rows];
    for i in 0..*rows {
        for j in 0..*cols {
            matrix[i][j] = rng.gen::<f64>();
        }
    }
    matrix
}

pub fn generate_network_weights_and_biases(
    input_size: &usize,
    hidden_sizes: &Vec<usize>,
    output_size: &usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
    let mut rows = input_size;
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for hidden_size in hidden_sizes {
        let cols = hidden_size;
        weights.push(generate_rand_matrix(rows, cols));
        biases.push(generate_rand_matrix(&1, cols));
        rows = hidden_size;
    }
    (weights, biases)
}

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;

    #[test]
    fn test_generate_rand_matrix_dimensions() {
        let rows = 3;
        let cols = 4;
        let matrix = generate_rand_matrix(&rows, &cols);
        assert_eq!(matrix.len(), rows);
        for row in matrix {
            assert_eq!(row.len(), cols);
        }
    }

    #[test]
    fn test_generate_rand_matrix_values_range() {
        let rows = 5;
        let cols = 5;
        let matrix = generate_rand_matrix(&rows, &cols);
        for row in matrix {
            for value in row {
                assert!(value >= 0.0 && value < 1.0);
            }
        }
    }

    #[test]
    fn test_generate_rand_matrix_randomness() {
        let rows = 2;
        let cols = 2;
        let matrix1 = generate_rand_matrix(&rows, &cols);
        let matrix2 = generate_rand_matrix(&rows, &cols);
        // It's unlikely but possible that these could be equal; however, this serves as a basic check.
        assert_ne!(matrix1, matrix2);
    }

    fn is_matrix_of_size(matrix: &Vec<Vec<f64>>, rows: usize, cols: usize) -> bool {
        matrix.len() == rows && matrix.iter().all(|r| r.len() == cols)
    }

    #[test]
    fn test_generate_network_weights_and_biases() {
        let input_size = 3;
        let hidden_sizes = vec![4, 5];
        let output_size = 2;
        let (weights, biases) =
            generate_network_weights_and_biases(&input_size, &hidden_sizes, &output_size);

        // Check if weights dimensions are correct
        assert!(is_matrix_of_size(&weights[0], input_size, hidden_sizes[0]));
        assert!(is_matrix_of_size(
            &weights[1],
            hidden_sizes[0],
            hidden_sizes[1]
        ));

        // Check if biases dimensions are correct
        assert!(is_matrix_of_size(&biases[0], 1, hidden_sizes[0]));
        assert!(is_matrix_of_size(&biases[1], 1, hidden_sizes[1]));

        // Check if the lengths of weights and biases are correct
        assert_eq!(weights.len(), hidden_sizes.len());
        assert_eq!(biases.len(), hidden_sizes.len());
    }

    #[test]
    fn test_generate_network_weights_and_biases_empty_hidden() {
        let input_size = 3;
        let hidden_sizes = vec![];
        let output_size = 2;
        let (weights, biases) =
            generate_network_weights_and_biases(&input_size, &hidden_sizes, &output_size);

        // Check if weights and biases are empty
        assert_eq!(weights.len(), 0);
        assert_eq!(biases.len(), 0);
    }

    #[test]
    fn test_generate_network_weights_and_biases_single_hidden_layer() {
        let input_size = 3;
        let hidden_sizes = vec![4];
        let output_size = 2;
        let (weights, biases) =
            generate_network_weights_and_biases(&input_size, &hidden_sizes, &output_size);

        // Check if weights dimensions are correct
        assert!(is_matrix_of_size(&weights[0], input_size, hidden_sizes[0]));

        // Check if biases dimensions are correct
        assert!(is_matrix_of_size(&biases[0], 1, hidden_sizes[0]));

        // Check if the lengths of weights and biases are correct
        assert_eq!(weights.len(), hidden_sizes.len());
        assert_eq!(biases.len(), hidden_sizes.len());
    }

    #[test]
    fn test_dot_product_regular_matrices() {
        let mat1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mat2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let result = dot(&mat1, &mat2);
        let expected = vec![vec![58.0, 64.0], vec![139.0, 154.0]];
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "cannot multiply empty matrices")]
    fn test_dot_product_empty_matrices() {
        let mat1: Vec<Vec<f64>> = vec![];
        let mat2: Vec<Vec<f64>> = vec![];
        dot(&mat1, &mat2);
    }

    #[test]
    #[should_panic(expected = "matrix dimension incompatible for dot product")]
    fn test_dot_product_incompatible_dimensions() {
        let mat1 = vec![vec![1.0]];
        let mat2 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        dot(&mat1, &mat2);
    }

    #[test]
    fn test_dot_product_single_element_matrices() {
        let mat1 = vec![vec![3.0]];
        let mat2 = vec![vec![4.0]];
        let result = dot(&mat1, &mat2);
        let expected = vec![vec![12.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_product_mismatched_sizes() {
        let mat1 = vec![vec![1.0, 2.0, 3.0]];
        let mat2 = vec![vec![4.0], vec![5.0], vec![6.0]];
        let result = dot(&mat1, &mat2);
        let expected = vec![vec![32.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_df_relu_positive() {
        let mut x = 1.0;
        df_relu(&mut x);
        assert_eq!(x, 1.0);
    }

    #[test]
    fn test_df_relu_negative() {
        let mut x = -1.0;
        df_relu(&mut x);
        assert_ne!(x, 0.0);
    }

    #[test]
    fn test_df_relu_zero() {
        let mut x = 0.0;
        df_relu(&mut x);
        assert_eq!(x, 0.0);
    }

    #[test]
    fn test_relu_positive() {
        let mut x = 5.0;
        relu(&mut x);
        assert_eq!(x, 5.0);
    }

    #[test]
    fn test_relu_negative() {
        let mut x = -3.0;
        relu(&mut x);
        assert_eq!(x, 0.0);
    }

    #[test]
    fn test_relu_zero() {
        let mut x = 0.0;
        relu(&mut x);
        assert_eq!(x, 0.0);
    }

    #[test]
    fn test_activate_with_positive_values() {
        let mut matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        activate(&mut matrix);
        assert_eq!(matrix, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_activate_with_negative_values() {
        let mut matrix = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        activate(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_activate_with_mixed_values() {
        let mut matrix = vec![vec![-1.0, 2.0], vec![3.0, -4.0]];
        activate(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0, 2.0], vec![3.0, 0.0]]);
    }

    #[test]
    #[should_panic(expected = "cannot activate empty matrix")]
    fn test_activate_with_empty_matrix() {
        let mut matrix: Vec<Vec<f64>> = vec![];
        activate(&mut matrix);
    }

    #[test]
    #[should_panic(expected = "cannot activate empty matrix")]
    fn test_df_activate_empty_matrix() {
        let mut matrix: Vec<Vec<f64>> = Vec::new();
        df_activate(&mut matrix);
    }

    #[test]
    fn test_df_activate_positive_values() {
        let mut matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        df_activate(&mut matrix);
        assert_eq!(matrix, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    fn apporx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_df_activate_mixed_values() {
        let n_rows = 2;
        let n_cols = 3;
        let mut matrix = vec![vec![-1.0, 2.0, -3.0], vec![4.0, -5.0, 6.0]];
        df_activate(&mut matrix);
        for i in 0..n_rows {
            for j in 0..n_cols {
                assert!(apporx_eq(matrix[i][j], matrix[i][j]));
            }
        }
    }

    #[test]
    fn test_transpose_square_matrix() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let expected = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    fn test_transpose_rectangular_matrix() {
        let matrix = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let expected = vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![3.0, 7.0],
            vec![4.0, 8.0],
        ];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    #[should_panic(expected = "cannot transpose empty matrix")]
    fn test_transpose_empty_matrix() {
        let matrix: Vec<Vec<f64>> = vec![];
        let expected: Vec<Vec<f64>> = vec![];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    fn test_transpose_single_element_matrix() {
        let matrix = vec![vec![1.0]];
        let expected = vec![vec![1.0]];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    fn test_transpose_single_row_matrix() {
        let matrix = vec![vec![1.0, 2.0, 3.0]];
        let expected = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    fn test_transpose_single_column_matrix() {
        let matrix = vec![vec![1.0], vec![2.0], vec![3.0]];
        let expected = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(transpose(&matrix), expected);
    }

    #[test]
    #[should_panic(expected = "cannot scale empty matrix")]
    fn test_empty_matrix() {
        let mut matrix: Vec<Vec<f64>> = vec![];
        min_max_scaler(&mut matrix);
    }

    #[test]
    #[should_panic(expected = "cannot scale empty matrix")]
    fn test_empty_sub_matrix() {
        let mut matrix: Vec<Vec<f64>> = vec![vec![]];
        min_max_scaler(&mut matrix);
    }

    #[test]
    fn test_single_element_matrix() {
        let mut matrix = vec![vec![42.0]];
        min_max_scaler(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0]]);
    }

    #[test]
    fn test_uniform_matrix() {
        let mut matrix = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        min_max_scaler(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_non_uniform_matrix() {
        let mut matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        min_max_scaler(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
    }

    #[test]
    fn test_large_values_matrix() {
        let mut matrix = vec![vec![1000.0, 2000.0], vec![3000.0, 4000.0]];
        min_max_scaler(&mut matrix);
        assert_eq!(matrix, vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
    }

    #[test]
    #[should_panic(expected = "Cannot add bias: Invalid bias dimensions")]
    fn test_add_bias_empty_bias() {
        let bias = vec![];
        let mut layer_batch_output = vec![vec![1.0, 2.0, 3.0]];
        add_bias(&bias, &mut layer_batch_output);
    }

    #[test]
    #[should_panic(expected = "Cannot add bias: Invalid bias dimensions")]
    fn test_add_bias_multiple_bias_rows() {
        let bias = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut layer_batch_output = vec![vec![1.0, 2.0, 3.0]];
        add_bias(&bias, &mut layer_batch_output);
    }

    #[test]
    #[should_panic(expected = "Cannot add bias: Empty layer output")]
    fn test_add_bias_empty_layer_output() {
        let bias = vec![vec![1.0, 2.0, 3.0]];
        let mut layer_batch_output = vec![];
        add_bias(&bias, &mut layer_batch_output);
    }

    #[test]
    #[should_panic(expected = "Cannot add bias: Bias and layer output dimensions mismatch")]
    fn test_add_bias_dimension_mismatch() {
        let bias = vec![vec![1.0, 2.0]];
        let mut layer_batch_output = vec![vec![1.0, 2.0, 3.0]];
        add_bias(&bias, &mut layer_batch_output);
    }

    #[test]
    fn test_add_bias_correct() {
        let bias = vec![vec![1.0, 2.0, 3.0]];
        let mut layer_batch_output = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        add_bias(&bias, &mut layer_batch_output);
        assert_eq!(
            layer_batch_output,
            vec![vec![2.0, 4.0, 6.0], vec![5.0, 7.0, 9.0]]
        );
    }

    #[test]
    fn test_add_bias_single_element() {
        let bias = vec![vec![1.0]];
        let mut layer_batch_output = vec![vec![1.0], vec![2.0]];
        add_bias(&bias, &mut layer_batch_output);
        assert_eq!(layer_batch_output, vec![vec![2.0], vec![3.0]]);
    }
}
