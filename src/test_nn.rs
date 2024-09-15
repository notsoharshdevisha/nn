use crate::utils::{
    activate,
    add_bias,
    batch_err,
    batch_err_per,
    df_activate,
    dot,
    generate_network_weights_and_biases,
    generate_rand_matrix,
    transpose,
};

fn train_batch(
    batch_inputs: &[Vec<f64>],
    batch_targets: &[Vec<f64>],
    network_weights: &mut Vec<Vec<Vec<f64>>>,
    network_biases: &mut Vec<Vec<Vec<f64>>>,
    learning_rate: f64,
    epoch: usize,
) {
    let (output, layer_outputs) = forward_pass(batch_inputs, network_weights, network_biases);
    let err = batch_err(&output, &batch_targets);
    let err_per = batch_err_per(&output, batch_targets);
    if epoch % 100 == 0 {
        println!("epoch => {}", epoch);
        println!("batch err per => {:?}", err_per);
    }
    back_propagate(
        &err,
        network_weights,
        network_biases,
        &layer_outputs,
        learning_rate,
    );
}

fn train(
    network_weights: &mut Vec<Vec<Vec<f64>>>,
    network_biases: &mut Vec<Vec<Vec<f64>>>,
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<Vec<f64>>,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
) {
    let batch_count = inputs.len() / batch_size;
    for epoch in 0..epochs {
        for i in 0..batch_count {
            let (start, end) = (i * batch_size, (i + 1) * batch_size);
            let batch_inputs = &inputs[start..end];
            let batch_targets = &targets[start..end];
            train_batch(
                batch_inputs,
                batch_targets,
                network_weights,
                network_biases,
                learning_rate,
                epoch,
            );
        }
    }
}

fn update_err(curr_err: &Vec<Vec<f64>>, curr_layer_weights: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let weights_transpose = transpose(&curr_layer_weights);
    dot(&curr_err, &weights_transpose)
}

fn update_weights(grad: &Vec<Vec<f64>>, weights: &mut Vec<Vec<f64>>, learning_rate: f64) {
    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            weights[i][j] -= learning_rate * grad[i][j];
        }
    }
}

fn update_biases(grad: &Vec<Vec<f64>>, bias: &mut Vec<Vec<f64>>, learning_rate: f64) {
    for i in 0..bias.len() {
        for j in 0..bias[i].len() {
            bias[i][j] -= learning_rate * grad[i][j];
        }
    }
}

fn back_propagate(
    err: &Vec<Vec<f64>>,
    network_weights: &mut Vec<Vec<Vec<f64>>>,
    network_biases: &mut Vec<Vec<Vec<f64>>>,
    network_outputs: &Vec<Vec<Vec<f64>>>,
    learning_rate: f64,
) {
    if network_weights.len() != network_biases.len() {
        panic!("length of network weights and biases cannot be different");
    }
    let mut curr_err = err.clone();
    for i in (0..network_weights.len()).rev() {
        let layer_input = transpose(&network_outputs[i]);
        // println!("layer => {}", i);
        // println!("    layer_input => {:?}", layer_input);
        // println!("    curr_err => {:?}", curr_err);
        let mut grad = dot(&layer_input, &curr_err);
        df_activate(&mut grad);
        // println!("    grad => {:?}", grad);
        // println!("    network_weights[i] => {:?}", network_weights[i]);
        update_weights(&grad, &mut network_weights[i], learning_rate);
        update_biases(&curr_err, &mut network_biases[i], learning_rate);
        curr_err = update_err(&curr_err, &network_weights[i]);
    }
}

fn forward_pass(
    batch_inputs: &[Vec<f64>],
    weights: &Vec<Vec<Vec<f64>>>,
    biases: &Vec<Vec<Vec<f64>>>,
) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
    if weights.len() != biases.len() {
        panic!("length of network weights and biases cannot be different")
    }
    let mut layer_outputs: Vec<Vec<Vec<f64>>> = vec![];
    let mut layer_output: Vec<Vec<f64>> = batch_inputs.to_vec();
    layer_outputs.push(layer_output.clone());
    for i in 0..weights.len() - 1 {
        layer_output = dot(&layer_output, &weights[i]);
        add_bias(&biases[i], &mut layer_output);
        activate(&mut layer_output);
        layer_outputs.push(layer_output.clone());
    }
    layer_output = dot(&layer_output, &weights[weights.len() - 1]);
    add_bias(&biases[biases.len() - 1], &mut layer_output);
    activate(&mut layer_output);
    return (layer_output, layer_outputs);
}

fn predict(
    inputs: &[Vec<f64>],
    weights: &Vec<Vec<Vec<f64>>>,
    biases: &Vec<Vec<Vec<f64>>>,
) -> Vec<Vec<f64>> {
    let (output, _) = forward_pass(inputs, weights, biases);
    output
}

#[cfg(test)]
mod test_test_nn {
    use super::*;

    #[test]
    #[should_panic(expected = "length of network weights and biases cannot be different")]
    fn test_forward_pass_mismatched_weights_biases() {
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![vec![vec![0.1, 0.2], vec![0.3, 0.4]]];
        let biases = vec![
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![vec![0.5, 0.6], vec![0.7, 0.8]],
        ];
        forward_pass(&inputs, &weights, &biases);
    }

    fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_forward_pass() {
        let inputs = vec![vec![1.0, 2.0]];
        let weights = vec![
            vec![vec![0.2, 0.8], vec![0.6, 0.4]],
            vec![vec![0.5, 0.9], vec![0.3, 0.7]],
        ];
        let biases = vec![vec![vec![0.1, 0.2]], vec![vec![0.3, 0.4]]];
        let (output, layer_outputs) = forward_pass(&inputs, &weights, &biases);
        let expected_output = vec![vec![1.59, 3.01]];
        let expected_layer_outputs = vec![
            vec![vec![1.0, 2.0]],
            vec![vec![1.5, 1.8]],
            vec![vec![1.59, 3.01]],
        ];
        let epsilon = 1e-6;
        assert_eq!(layer_outputs.len(), weights.len());
        for (layer, expected_layer) in layer_outputs.iter().zip(expected_layer_outputs.iter()) {
            for (neuron, expected_neuron) in layer.iter().zip(expected_layer.iter()) {
                for (value, expected_value) in neuron.iter().zip(expected_neuron.iter()) {
                    assert!(approx_equal(*value, *expected_value, epsilon));
                }
            }
        }
        for (output_neuron, expected_output_neuron) in output.iter().zip(expected_output.iter()) {
            for (value, expected_value) in output_neuron.iter().zip(expected_output_neuron.iter()) {
                assert!(approx_equal(*value, *expected_value, epsilon));
            }
        }
    }

    #[test]
    #[should_panic(expected = "length of network weights and biases cannot be different")]
    fn test_back_propagate_weights_biases_length_mismatch() {
        let err = vec![vec![0.1, 0.2]];
        let mut network_weights = vec![vec![vec![0.5, 0.3]]];
        let mut network_biases = vec![vec![vec![0.1]], vec![vec![0.1]]];
        let network_outputs = vec![vec![vec![0.6, 0.9]]];
        let learning_rate = 0.05;
        back_propagate(
            &err,
            &mut network_weights,
            &mut network_biases,
            &network_outputs,
            learning_rate,
        );
    }

    #[test]
    fn test_back_propagate_proper_execution() {
        let err = vec![vec![0.1, 0.2]];
        let mut network_weights = vec![vec![vec![0.5, 0.3], vec![0.4, 0.2]]];
        let mut network_biases = vec![vec![vec![0.1, 0.1]]];
        let network_outputs = vec![vec![vec![0.6, 0.9]]];
        let weights_before = network_weights.clone();
        let learning_rate = 0.05;
        back_propagate(
            &err,
            &mut network_weights,
            &mut network_biases,
            &network_outputs,
            learning_rate,
        );
        assert!(weights_before != network_weights);
        assert_eq!(network_weights.len(), 1);
        assert_eq!(network_weights[0].len(), 2);
        assert_eq!(network_weights[0][0].len(), 2);
        assert_eq!(network_biases.len(), 1);
        assert_eq!(network_biases[0].len(), 1);
        assert_eq!(network_biases[0][0].len(), 2);
    }

    #[test]
    fn test_train_single_input_proper_execution() {
        let mut network_weights = vec![
            vec![vec![0.2, 0.8], vec![0.6, 0.4]],
            vec![vec![0.5, 0.9], vec![0.3, 0.7]],
            vec![vec![0.3], vec![0.2]],
        ];
        let mut network_biases = vec![vec![vec![0.1, 0.2]], vec![vec![0.3, 0.4]], vec![vec![0.5]]];
        let inputs = vec![vec![1.0, 2.0]];
        let targets = vec![vec![0.0]];
        let learning_rate = 0.01;
        let epochs = 1000;
        let batch_size = 1;
        let weights_before = network_weights.clone();
        let biases_before = network_biases.clone();
        // println!("weights_before => {:?}", weights_before);
        // println!("biases_before => {:?}", biases_before);
        train(
            &mut network_weights,
            &mut network_biases,
            &inputs,
            &targets,
            learning_rate,
            epochs,
            batch_size,
        );
        // println!("weights_after => {:?}", network_weights);
        // println!("biases_after => {:?}", network_biases);
        assert_ne!(weights_before, network_weights);
        assert_ne!(biases_before, network_biases);
    }

    #[test]
    #[should_panic(expected = "length of network weights and biases cannot be different")]
    fn test_train_mismatched_weights_biases() {
        let mut network_weights = vec![vec![vec![0.2, 0.8], vec![0.6, 0.4]]];
        let mut network_biases = vec![vec![vec![0.1, 0.2]], vec![vec![0.3, 0.4]]];
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let learning_rate = 0.01;
        let epochs = 1;
        let batch_size = 2;

        train(
            &mut network_weights,
            &mut network_biases,
            &inputs,
            &targets,
            learning_rate,
            epochs,
            batch_size,
        );
    }

    #[test]
    fn test_convergence() {
        let mut network_weights = vec![
            vec![vec![0.2, 0.8], vec![0.6, 0.4]],
            vec![vec![0.3], vec![0.2]],
        ];
        let mut network_biases = vec![vec![vec![0.1, 0.2]], vec![vec![0.5]]];
        let inputs = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
            vec![7.0, 7.0],
            vec![8.0, 8.0],
        ];
        let targets = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
        ];
        let learning_rate = 0.0001;
        let epochs = 50000;
        let batch_size = 4;
        let weights_before = network_weights.clone();
        let biases_before = network_biases.clone();
        println!("weights_before => {:?}", weights_before);
        println!("biases_before => {:?}", biases_before);
        train(
            &mut network_weights,
            &mut network_biases,
            &inputs,
            &targets,
            learning_rate,
            epochs,
            batch_size,
        );
        println!("weights_after => {:?}", network_weights);
        println!("biases_after => {:?}", network_biases);
        assert_ne!(weights_before, network_weights);
        assert_ne!(biases_before, network_biases);
        let predictions: Vec<Vec<f64>> = predict(&inputs, &network_weights, &network_biases);
        println!("predictions => {:?}", predictions);
        for i in 0..predictions.len() {
            assert!(approx_equal(predictions[i][0], targets[i][0], 1e-1));
        }
    }
}
