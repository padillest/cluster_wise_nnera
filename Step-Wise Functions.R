source("kNN-ERA Functions.R")

set.seed(1234)

#### Main Function #### 
compute_step_wise <- function(k, x, y,
                              x_per_lv,
                              hdn_per_lv,
                              hdn_layer_neurons,
                              lv_neurons,
                              output_neurons,
                              iterations,
                              num_y) { 
  
  #' Computes a step-wise k-means and NN-ERA procedure.

  
  # Storage 
  output_values <- list()
  
  # Compute k-means
  k_means_output <- kmeans(x, centers = k)
  
  # Subset the data into the respective clusters 
  for (i in 1:k) { 
    
    cluster_idx <- which(k_means_output[[1]] == i)
    cluster_data <- x[cluster_idx, ]
    cluster_output <- y[cluster_idx, ]
    
    # Performs NN-ERA on the subset
    output <- compute_nn_era(x_per_lv = x_per_lv,
                             hdn_per_lv = hdn_per_lv,
                             hdn_layer_neurons = hdn_layer_neurons,
                             lv_neurons = lv_neurons,
                             output_neurons = output_neurons,
                             x = cluster_data, y = cluster_output, 
                             num_y = num_y,
                             iterations = iterations)
    output_values <- append(output_values, output)
  }
  return(list(k_means_output, output_values))
}

#### compute_nn_era Function #### 
compute_nn_era <- function(x_per_lv, hdn_per_lv, hdn_layer_neurons,
                           lv_neurons, output_neurons,
                           x, y, num_y, iterations) {
  #' Computes the NN-ERA using functions within the script.
  
  # Initialize all weights 
  initial_weights <- initialize_all_weights(x_per_lv = x_per_lv,
                                            hdn_per_lv = hdn_per_lv,
                                            num_y = num_y)
  
  wh <- initial_weights[[1]]
  wl <- initial_weights[[2]]
  wout <- initial_weights[[3]]
  wh_non_zero <- initial_weights[[4]]
  wl_non_zero <- initial_weights[[5]]
  
  # Initialize all biases 
  initial_biases <- initialize_all_biases(X = x, 
                                          hdn_layer_neurons = hdn_layer_neurons,
                                          lv_neurons = lv_neurons,
                                          output_neurons = output_neurons)
  bias_hidden <- initial_biases[[1]]
  bias_latent <- initial_biases[[2]]
  bias_output <- initial_biases[[3]]
  
  # Iteration loop 
  for (i in 1:iterations) { 
    # Compute forward propagation 
    fprop <- compute_fprop(X = x,
                           wh = wh,
                           bias_hidden = bias_hidden,
                           wl = wl, 
                           bias_latent = bias_latent, 
                           wout = wout, 
                           bias_out = bias_output)
    
    # Compute error 
    error <- compute_error(y_matrix = y,
                           y_predicted_matrix = fprop[[3]])
    
    # Compute MSE 
    
    mse <- compute_mse(y_matrix = y, 
                       y_predicted_matrix = fprop[[3]])
    
    relative_bias <- compute_relative_bias(y_matrix = y, 
                                           y_predicted_matrix = fprop[[3]])
    
    # TODO intercept estimates 
    intercept_estimates <- estimate_intercept(Y = y,
                                              estimate = fprop[[3]],
                                              cluster_idx = 1:nrow(y))
    
    # Compute activation derivatives 
    derivatives <- compute_derivatives(fprop)
    
    # Compute back propagation 
    bprop <- compute_bprop(error = error, 
                           derivatives = derivatives,
                           wout = wout, 
                           wl = wl)
    
    # Update weights 
    updated_weights <- update_all_weights(wh = wh, 
                                          wl = wl, 
                                          wout = wout, 
                                          X = x,
                                          fprop = fprop, 
                                          bprop = bprop,
                                          wh_non_zero = wh_non_zero,
                                          wl_non_zero = wl_non_zero,
                                          learning_rate = learning_rate)
    wh <- updated_weights[[1]]
    wl <- updated_weights[[2]]
    wout <- updated_weights[[3]]
    
    # Update biases
    updated_biases <- update_all_biases(bias_hidden = bias_hidden, 
                                        bias_latent = bias_latent,
                                        bias_output = bias_output, 
                                        bprop = bprop,
                                        HN_per_lv = hdn_per_lv,
                                        learning_rate = learning_rate)
    bias_hidden <- updated_biases[[1]]
    bias_latent <- updated_biases[[2]]
    bias_output <- updated_biases[[3]]
  
  }
  
  mse_output <- sum(mse)/nrow(mse)
  relative_bias_output <- sum(relative_bias)/nrow(relative_bias)
  
  return(list(error, wout, intercept_estimates, mse_output, fprop[[3]], relative_bias_output))
}

#### update_all_biases Function #### 

update_all_biases <- function(bias_hidden, 
                              bias_latent,
                              bias_output,
                              bprop,
                              HN_per_lv,
                              learning_rate) { 

  bias_hidden <- update_bias(layer = "hidden",
                             bias_matrix = bias_hidden,
                             layer_derivative_matrix = bprop[[1]],
                             learning_rate = learning_rate,
                             hdn_node_per_lv = HN_per_lv) 
  
  bias_latent <- update_bias(layer = "latent",
                             bias_matrix = bias_latent,
                             layer_derivative_matrix = bprop[[2]],
                             learning_rate = learning_rate)
  
  
  bias_output <- update_bias(layer = "output",
                             bias_matrix = bias_output,
                             layer_derivative_matrix = bprop[[3]],
                             learning_rate = learning_rate)
  
  return(list(bias_hidden,
              bias_latent,
              bias_output))
}

#### update_all_weights Function #### 

update_all_weights <- function(wh, wl, wout, X,
                               fprop, bprop, 
                               wh_non_zero,
                               wl_non_zero,
                               learning_rate) {
  
  
  wh <- update_weights(layer = "hidden", 
                       weight_matrix = wh, 
                       input_matrix = X,
                       derivative_input_matrix = bprop[[1]],
                       learning_rate = learning_rate,
                       non_zero_weight_indices = wh_non_zero)
  
  wl <- update_weights(layer = "latent", 
                       weight_matrix = wl, 
                       input_matrix = fprop[[1]],
                       derivative_input_matrix = bprop[[2]],
                       learning_rate = learning_rate,
                       non_zero_weight_indices = wl_non_zero) 
  
  wout <- update_weights(layer = "output", 
                         weight_matrix = wout, 
                         input_matrix = fprop[[2]],
                         derivative_input_matrix = bprop[[3]],
                         learning_rate = learning_rate) 
  
  return(list(wh, wl, wout))
}

  

#### compute_derivatives Function #### 

compute_derivatives <- function(data) {
  
  slope_output_layer <- activation_derivative(layer_matrix = data[[3]],
                                              activation_function = "linear")
  
  slope_latent_layer <- activation_derivative(layer_matrix = data[[2]], 
                                              activation_function = "linear")
  
  slope_hidden_layer <- activation_derivative(layer_matrix = data[[1]],
                                              activation_function = "sigmoid")
  
  return(list(slope_hidden_layer, 
              slope_latent_layer,
              slope_output_layer))
}

#### compute_bprop Function ####

compute_bprop <- function(error, 
                          derivatives,
                          wout, wl) { 
  
  d_output <- back_prop(layer = "output",
                        derivative_input_matrix = error,
                        layer_derivative_matrix = derivatives[[3]]) 
  
  d_latent <- back_prop(layer = "latent",
                        derivative_input_matrix = d_output,
                        layer_derivative_matrix = derivatives[[2]],
                        weight_matrix = wout)
  
  d_hidden <- back_prop(layer = "hidden",
                        derivative_input_matrix = d_latent,
                        layer_derivative_matrix = derivatives[[1]],
                        weight_matrix = wl)
  return(list(d_hidden,
              d_latent,
              d_output))
}

#### compute_fprop Function #### 

compute_fprop <- function(X, wh, bias_hidden, 
                          wl, bias_latent, 
                          wout, bias_out) { 
  
  # Hidden Layer # 
  hidden_layer_input <- forward_prop(layer = "hidden",
                                     input_matrix = X,
                                     weight_matrix = wh,
                                     layer_bias = bias_hidden)
  
  hidden_layer_output <- activation(layer_matrix = hidden_layer_input,
                                    activation_function = "sigmoid")
  
  # Latent Layer # 
  latent_layer_input <- forward_prop(layer = "latent", 
                                     input_matrix = hidden_layer_output, 
                                     weight_matrix = wl, 
                                     layer_bias = bias_latent)
  
  latent_layer_output <- activation(layer_matrix = latent_layer_input,
                                    activation_function = "linear")
  
  # Output Layer # 
  output_layer_input <- forward_prop(layer = "output",
                                     input_matrix = latent_layer_output, 
                                     weight_matrix = wout,
                                     layer_bias = bias_out)
  
  output_layer_output <- activation(layer_matrix = output_layer_input,
                                    activation_function = "linear")
  
  return(list(hidden_layer_output, 
              latent_layer_output,
              output_layer_output))
}


#### initialize_all_bias Function #### 

initialize_all_biases <- function(X,
                                  hdn_layer_neurons,
                                  lv_neurons,
                                  output_neurons) { 
  
  bias_hidden <- initialize_bias(X = X, 
                                 neuron_num = hdn_layer_neurons)
  
  bias_latent <- initialize_bias(X = X,
                                 neuron_num = lv_neurons)
  
  bias_out <- initialize_bias(X = X, 
                              neuron_num = output_neurons)
  
  return(list(bias_hidden,
              bias_latent,
              bias_out))
}

#### initialize_all_weights Function #### 
initialize_all_weights <- function(x_per_lv,
                                   hdn_per_lv,
                                   num_y) { 
  
  initial_w <- initialize_model_weights(layer_input = x_per_lv,
                                        layer_output = hdn_per_lv,
                                        num_Y = num_y)
  
  wh <- initial_w$Z_0[[1]]
  wl <- initial_w$L_0[[1]]
  wout <- initial_w$A_0
  
  wh_non_zero <- initial_w$Z_0[[2]]
  wl_non_zero <- initial_w$L_0[[2]]
  
  return(list(wh, wl, wout,
              wh_non_zero,
              wl_non_zero))
}

