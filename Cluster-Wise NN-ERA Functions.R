library(SimDesign)

#### Convert Assignment Vector #### 
convert_assignment_vec <- function(output) { 
  temp <- as.matrix(output)
  sw_assign_mat <- matrix(data = 0, 
                          nrow = nrow(temp),
                          ncol = length(unique(temp)))
  
  for (i in 1:nrow(temp)) { 
    cluster <- temp[i]
    sw_assign_mat[i, cluster] <- 1
  }
  
  return(sw_assign_mat)
}

#### Compute Relative Bias #### 

compute_relative_bias <- function(y_matrix, 
                                  y_predicted_matrix) { 
  
  relative_bias <- (y_predicted_matrix - y_matrix)/y_matrix
  
  return(relative_bias)
}

#### Compute Error Variance #### 

compute_mse <- function(y_matrix,
                        y_predicted_matrix) { 

  mse <- (y_matrix - y_predicted_matrix)^2
  return(mse)
}

#### Compute Classification Rate ####

compute_classification_rate <- function(output, 
                                        proportion) { 
  
  assignment <- output
  total_row <- nrow(assignment)
  
  if (proportion == "50") {
    a <- colSums(assignment[1:(total_row*0.5), ])[1]
    b <- colSums(assignment[1:(total_row*0.5), ])[2]
    if (a > b) { 
      c <- colSums(assignment[-(1:(total_row*0.5)), ])[2]
      output <- (a+c)/total_row
    } else { 
      c <- colSums(assignment[-(1:(total_row*0.5)), ])[1]
      output <- (b+c)/total_row
      
    }
    
  } else { 
    a <- colSums(assignment[1:(total_row*0.75), ])[1]
    b <- colSums(assignment[1:(total_row*0.75), ])[2]
    if (a > b) { 
      c <- colSums(assignment[-(1:total_row*0.75), ])[2]
      output <- (a+c)/total_row
      
    } else { 
      c <- colSums(assignment[-(1:total_row*0.75), ])[1]
      output <- (b+c)/total_row
    }
  }
  return(output * 100)
}

#### Generate Data #### 

generate_data <- function(int_mat, w_mat, a_mat, 
                          x_sig_mat, y_sig_mat, 
                          k_clusters, cluster_size,
                          cluster_mean) { 
  #' Generates a dataframe of clustered data according to 
  #' input matrices and cluster means.
  #' 
  #' Dependencies: 
  #'   SimDesign::rmvnorm
  #' 
  #' Parameters: 
  #'   int_mat: list 
  #'     A list of cluster intercepts. 
  #'     
  #'   w_mat: list
  #'     A list of weights matrices for each cluster. 
  #'     
  #'   a_mat: list
  #'     A list of square matrices for each cluster.
  #'     
  #'   x_sig_mat: list
  #'     A list of cluster variance-covariance matrices.
  #'     
  #'   y_sig_mat: list
  #'     A list of variance matrices for each cluster. 
  #'     
  #'   k_clusters: int 
  #'     The number of clusters to generate.
  #'     
  #'   cluster_size: list 
  #'     A list of the size of each ith cluster.
  #'   
  #'   cluster_mean: list
  #'     A list of the mean of each ith cluster. 
  
  # Error handling
  if ((length(cluster_size) != k_clusters) 
      | (length(int_mat) != k_clusters) 
      | (length(cluster_mean) != k_clusters) 
      | (length(w_mat) != k_clusters) 
      | (length(a_mat) != k_clusters)
      | (length(x_sig_mat) != k_clusters) 
      | (length(y_sig_mat) != k_clusters)) { 
    stop("The number of cluster sizes, cluster means, and the number of 
         clusters must be equal.")
    }
  
  # Initialize variables
  x_data <- NULL
  y_data <- NULL
  
  # Iterate through the number of clusters to generate data
  for (i in 1:k_clusters) {
    
    # Generate cluster X 
    cluster_x <- SimDesign::rmvnorm(n = cluster_size[[i]],
                                    mean = rep(x = cluster_mean[[i]],
                                               times = nrow(x_sig_mat[[i]])),
                                    sigma = x_sig_mat[[i]])
    
    # Generate cluster Y 
    cluster_y <- matrix(nrow = cluster_size[[i]],
                        ncol = nrow(x = y_sig_mat[[i]]))
    
    for (data in 1:cluster_size[[i]]) {
      cluster_y[data, ] <- SimDesign::rmvnorm(n = 1,
                                   mean = (int_mat[[i]] + 
                                             cluster_x[data, ] %*% 
                                             w_mat[[i]] %*% 
                                             a_mat[[i]]),
                                   sigma = y_sig_mat[[i]])
    }
    
    # Combine cluster data
    x_data <- rbind(x_data, cluster_x) 
    y_data <- rbind(y_data, cluster_y)
    
  }
  
  # Compute dataframe
  data <- data.frame(y_data, x_data)
  
  # Rename columns
  names(data) <- c(paste0("Y", 1:ncol(a_mat[[1]])),
                   paste0("X", 1:nrow(w_mat[[1]])))
  
  return(data)
   
}

#### Weights #### 

# Initialize Weights Function # 
initialize_weights <- function(input_array, 
                               output_array) { 
  #' Creates an initial matrix to maintain the ERA 
  #' structure. 
  #' 
  #' Parameters: 
  #'   input_array: vector
  #'     A vector of dimensions one by the number of 
  #'     latent variables.
  #'     
  #'  output_array: vector
  #'    A vector of dimensions one by the number of 
  #'    latent variables.
  #'
  #' Return:
  #'   A weight matrix that considers the 
  #'   ERA structure.
  
  # Error Handling
  if (length(input_array) != length(output_array)) { 
    stop("Inputs should have the same number of latent variables.")
  }
  
  # Storing the number of latent variables 
  lvs <- length(input_array)
  
  # Storing the total number of inputs and outputs
  total_input <- sum(input_array)
  total_output <- sum(output_array)
  
  # Initializing a 0 matrix with the dimensions of the total 
  # observed input and the total number of output
  weight_matrix <- matrix(data = 0,
                          nrow = total_input,
                          ncol = total_output)
  
  # Form component matrix with non-zero entries
  
  ## Initialize variables 
  row_idx <- 1 
  col_idx <- 1 
  temp_row <- 0
  temp_col <- 0
  
  ## Loop through the elements of the array
  for (i in 1:lvs) { 
    
    # Update the temporary variables 
    temp_row <- temp_row + input_array[i]
    temp_col <- temp_col + output_array[i]
    
    # Update the entries in the matrix 
    weight_matrix[row_idx:temp_row, col_idx:temp_col] <- 99
    
    # Update matrix indices
    row_idx <- row_idx + temp_row
    col_idx <- col_idx + temp_col
    
  }
  
  # Inputting random values in non-zero entries 
  non_zero_idx <- which(weight_matrix == 99)
  
  # Generating a randomized array of the 
  # length of non_zero entries
  random_weight_array <- rnorm(length(non_zero_idx))
  
  # Producing an output matrix with random entries 
  output_matrix_array <- weight_matrix 
  output_matrix_array[non_zero_idx] <- random_weight_array
  
  return(list(output_matrix_array, non_zero_idx))
  
}

# End of Initialize Weights Function #

# Initialize Model Weights Function # 

initialize_model_weights <- function(layer_input, 
                                     layer_output, 
                                     num_Y) {
  #' Initializes the weight matrix for its 
  #' respective layer.
  #' 
  #' Parameters: 
  #'   layer_input: vector
  #'     A vector of dimensions one by the number 
  #'     of nodes in the current layer. 
  #'     
  #'   layer_output: vector 
  #'     A vector of dimensions one by the number
  #'     of nodes in the next layer.
  #'     
  #'   num_Y: vector 
  #'     A vector of dimensions one by Y. 
  #'     
  #' Return: 
  #'   Weight matrix for respective row.
  
  # Error Handling 
  if (length(layer_input) != length(layer_output)) { 
    stop("Inputs should have the same number of latent variables.")
  }
  
  # Creating an output array
  Y_array <- rep(1, length(layer_output))
  
  # Initializing weights for input layer to hidden layer
  Z_0 <- initialize_weights(input_array  = layer_input, 
                            output_array = layer_output)
  
  # Initializing weights from hidden layer to latent variables 
  L_0 <- initialize_weights(input_array  = layer_output, 
                            output_array = Y_array)
  
  # Initializing weights from latent variable to output 
  A_0 <- matrix(data = rnorm(n = length(layer_input) * num_Y),
                nrow = length(layer_input),
                ncol = num_Y)
  
  return(list(Z_0 = Z_0, 
              L_0 = L_0,
              A_0 = A_0))
  
}

# End of Initialize Weights Function # 

# Update Weights Function # 

update_weights <- function(layer, weight_matrix,
                           non_zero_weight_indices,
                           input_matrix,
                           derivative_input_matrix,
                           learning_rate) {
  #' Updates the weight matrix for the respective layer. 
  #' 
  #' Parameters: 
  #'   layer: str, 'hidden', 'latent', 'output'
  #'     String that denotes the layer of the model. 
  #'     
  #'   weight_matrix: matrix
  #'     Matrix of dimensions of the nodes of the 
  #'     given layer by the number of latent variables. 
  #'     
  #'   non_zero_weight_indices: vector
  #'     Vector of the indices of non-zero values to 
  #'     maintain the NN-ERA structure.
  #'     
  #'   input_matrix: matrix 
  #'     Matrix of dimensions of N by the number of 
  #'     latent variables. 
  #'     
  #'   derivative_input_matrix: matrix
  #'     Matrix of dimensions N by the number of 
  #'     latent variables.
  #'     
  #'   learning_rate: int 
  #'     Integer that represents the step size. 
  #'     
  #' Return: 
  #'   Updated weight matrix according to the forward-backward propagation.
  
  # Layer check 
  if (layer == 'hidden') { 
    
    # Create a temporary matrix
    temp <- (t(input_matrix) %*% derivative_input_matrix) * learning_rate
    
    # Maintain the NN-ERA structure
    temp[-non_zero_weight_indices] <- 0 
    
    # Update weight matrix 
    weight_matrix <- weight_matrix + temp
    
  } else if (layer == 'latent') { 
    
    temp <- (t(input_matrix) %*% derivative_input_matrix) * learning_rate
    
    temp[-non_zero_weight_indices] <- 0 
    
    weight_matrix <- weight_matrix + temp
      
  } else if (layer == 'output') { 
      
    weight_matrix <- weight_matrix + (t(input_matrix) %*% derivative_input_matrix) * learning_rate
    
  }
  
  return(weight_matrix)
  
}

# End of Update Weights Function # 

#### Biases ####

# Initialize Bias Function # 

initialize_bias <- function(X, neuron_num) {
  #' Initializes the bias matrix for the respective layer.
  #' 
  #' Parameter: 
  #' 
  #'   X: matrix 
  #'     Matrix of dimensions N by the number of 
  #'     variables.
  #'     
  #'   neuron_num: int
  #'     Integer of the number of neurons in the layer.
  #' 
  #' Return: 
  #'   The bias matrix for the respective layer.
  
  # Initialize bias matrix 
  bias <- runif(n = neuron_num)
  
  # Populate bias matrix 
  bias_matrix <- matrix(data = rep(x = bias, 
                                   each = nrow(X)),
                        nrow = nrow(X),
                        byrow = FALSE)
  
  return(bias_matrix)
}

# End of Initialize Bias Function # 

# Update Bias Function # 
update_bias <- function(layer, bias_matrix,
                        layer_derivative_matrix,
                        learning_rate,
                        hdn_node_per_lv) { 
  #' Updates bias matrix of a given layer. 
  #' 
  #' Parameters: 
  #'   layer: str, 'hidden', 'latent', 'output'
  #'     String denoting the layer.
  #'  
  #'   bias_matrix: matrix 
  #'     A matrix of dimensions current layer nodes 
  #'     and next layer nodes. 
  #'     
  #'   layer_derivative_matrix: matrix
  #'     A matrix of dimensions of the layers and nodes.
  #'     
  #'   learning_rate: int
  #'     An integer that represents the step size. 
  #'     
  #'   hdn_node_per_lv: vector
  #'     A vector 

  # Layer check 
  if (layer == 'hidden') { 
    
    # Creating a zero matrix
    sum <- matrix(data = 0, 
                  nrow = nrow(layer_derivative_matrix), 
                  ncol = length(hdn_node_per_lv))
    
    # Initialize variables
    lower_col_range <- 1 
    upper_col_range <- 0 
    
    for (i in 1:ncol(sum)) {
      
      # Initialize j has the number of hidden nodes of the array
      upper_col_range <- upper_col_range + hdn_node_per_lv[i]
      
      # Update the bias matrix with respects to the learning rate
      bias_matrix[, lower_col_range:upper_col_range] <- bias_matrix[, lower_col_range:upper_col_range] +
        mean(rowSums(layer_derivative_matrix[, lower_col_range:upper_col_range])) * learning_rate
      
      # Increment the lower column range 
      lower_col_range <- upper_col_range + 1
      
    }
    
  } else if (layer == 'latent') { 
    
    # Update bias matrix
    bias_matrix <- bias_matrix + (mean(x = layer_derivative_matrix) * learning_rate)
      
  } else if (layer == 'output') { 
      
    # Update bias matrix 
    bias_matrix <- bias_matrix + mean(x = rowSums(x = layer_derivative_matrix)) * learning_rate
    
  }
  return(bias_matrix)
}

# End of Update Bias Function #

#### Activation ####

# Activation Function # 

activation <- function(layer_matrix, 
                       activation_function) { 
  #' Parses the activation function at a given layer. 
  #' 
  #' Parameters: 
  #'   layer_matrix: matrix 
  #'     A matrix of information of the previous layer.
  #'     
  #'   activation_function: str, 'sigmoid', 'linear', 'tanh', 'prelu'
  #'     String denoting the type of activation function. 
  #'     
  #' Return: 
  #'   Result of the layer matrix and the activation function.
  
  # Parsing the activation function type 
  if (activation_function == 'sigmoid') { 
    # Compute output
    output <- 1/(1 + exp(-layer_matrix))
    
  } else if (activation_function == 'linear') { 
    output <- layer_matrix
    
  } else if (activation_function == 'tanh') { 
    output <- tanh(layer_matrix)
    
  } else if (activation_function == 'prelu') { 
    output <- ifelse(test = layer_matrix < 0, 
                     yes  = prelu.alpha * layer_matrix,
                     no   = layer_matrix)
    
  }
  
  return(output)
  
}

# End of Activation Function # 

# Activation Derivative Function # 

activation_derivative <- function(layer_matrix, 
                                  activation_function) { 
  #' Computes the derivative of the activation layer 
  #' at its respective layer.
  #' 
  #' Parameters: 
  #'   layer_matrix: matrix 
  #'     A matrix of information from the previous layer.
  #'     
  #'   activation_function: str, 'sigmoid', 'linear', 'tanh', 'prelu'
  #'     String that represents the type of activation function.
  #'
  #' Return:
  #'   Matrix of dimensions N by the number of nodes at 
  #'   the given layer. 
  
  # Parse the type of activation function
  if (activation_function == 'sigmoid') { 
    # Compute the derivative of the activation function
    output <- layer_matrix * (1 - layer_matrix) 
    
  } else if (activation_function == 'linear') { 
    output <- matrix(data = 1,
                     nrow = dim(x = layer_matrix)[1],
                     ncol = dim(x = layer_matrix)[2])
    
    
  } else if (activation_function == 'tanh') { 
    output <- 1 - (layer_matrix^2)
    
  } else if (activation_function == 'prelu') { 
    output <- ifelse(test = layer_matrix < 0, 
                     yes = prelu.alpha,
                     no = 1)
    
  }
  
  return(output)
  
}

# End of Activation Derivative Function # 

#### Forward Propagation #### 

# Forward Propagation Function # 

forward_prop <- function(layer,
                         input_matrix,
                         weight_matrix,
                         layer_bias) {
  #' Computes forward propagation according to the inputs,
  #' weights, and bias.
  #' 
  #' Parameters: 
  #'   layer: str, 'hidden', 'latent', 'output'
  #'     String denoting the layer of the model.
  #'     
  #'   input_matrix: matrix
  #'     Matrix of dimensions N by X. 
  #'     
  #'   weight_matrix: matrix 
  #'     Matrix of dimensions of the number of nodes at current 
  #'     layer by the number of nodes at the next layer. 
  #'     
  #'   layer_bias: matrix
  #'     Matrix of dimensions N by the number of latent variables.
  #'   
  #' Return: 
  #'   Matrix of dimensions N by the number of nodes at the 
  #'   next layer. 
  
  # Layer check 
  if (layer == 'hidden') { 
    
    layer_output <- scale(x = ((input_matrix %*% weight_matrix) + layer_bias),
                         center = FALSE, 
                         scale = TRUE) 
    
  } else if (layer == 'latent') { 
    
    layer_output <- scale(x = ((input_matrix %*% weight_matrix) + layer_bias), 
                          center = FALSE,
                          scale = TRUE)
      
  } else if (layer == 'output') { 
    
    layer_output <- (input_matrix %*% weight_matrix) + layer_bias
      
  }
  
  return(layer_output)
}

# End of Forward Propagation Function # 


#### Back Propagation #### 

back_prop <- function(layer, 
                      derivative_input_matrix,
                      layer_derivative_matrix,
                      weight_matrix) { 
  #' Computes back propagation of the network.
  #' 
  #' Parameters: 
  #'   layer: str, 'hidden', 'latent', 'output'
  #'     String denoting the layer of the model. 
  #'     
  #'   derivative_input_matrix: matrix
  #'     Matrix of dimensions N by the number of latent variables.
  #'     
  #'   layer_derivative_matrix: matrix
  #'     Matrix of dimensions N by the number of latent variables.
  #'     
  #'   weight_matrix: matrix
  #'     Matrix of dimensions of the number of nodes at the given 
  #'     layer by the number of latent variables. 
  #'     
  #' Return: 
  #'   Matrix of dimensions N by the number of latent variables. 
  
  # Layer check 
  if (layer == 'hidden') { 
    
    output <- (derivative_input_matrix %*% t(weight_matrix)) * 
      layer_derivative_matrix
    
  } else if (layer == 'latent') { 
      
    output <- (derivative_input_matrix %*% t(weight_matrix)) * 
      layer_derivative_matrix
    
  } else if (layer == 'output') { 
      
    output <- derivative_input_matrix * layer_derivative_matrix 
    
  }
  
  return(output)
}

#### Error #### 

# Compute Error Function # 

compute_error <- function(y_matrix,
                          y_predicted_matrix) { 
  #' Computes the error matrix after feed forward propagation. 
  #' 
  #' Parameters: 
  #'   y_matrix: matrix 
  #'     Matrix of dimensions N by the number of 
  #'     latent variables. 
  #'  
  #'   y_predicted_matrix: matrix 
  #'     Matrix of dimensions N by the number of 
  #'     latent variables.
  #'     
  #' Return: 
  #'   Error matrix of dimensions N by the number of 
  #'   latent variables.
  
  # Compute error matrix 
  error_matrix <- y_matrix - y_predicted_matrix
  
  return (error_matrix)
  }

# End of Compute Error Function # 

#### Cluster Centroids #### 

# Initialize Cluster Centroids Function #

initialize_centroids <- function(k_clusters,
                                 error_matrix) {
  #' Initializes the matrix of cluster centroids. 
  #' 
  #' Parameters: 
  #'   k_clusters: int 
  #'     Integer representing the number of clusters. 
  #'     
  #'   error_matrix: matrix 
  #'     Matrix of dimensions N by the number of 
  #'     latent variables. 
  #'     
  #' Return: 
  #'   A centroid matrix of dimensions K by the 
  #'   number of latent variables.

  # Initialize zero matrix of dimensions K by the 
  # number of latent variables. 
  centroid_matrix <- matrix(data = 0, 
                            nrow = k_clusters,
                            ncol = ncol(error_matrix))
  
  # Randomly select K number of points in the error matrix as the initial
  # centroids
  
  for (i in 1:k_clusters) { 
    
    centroid_matrix[i, ] <- error_matrix[sample(x = nrow(error_matrix),
                                                size = 1), ]
    
  }
  
  
  
  return(centroid_matrix)
  
}

# End of Initialize Cluster Centroids Function # 

# Update Cluster Centroids Function # 

update_centroids <- function(error_matrix, 
                             centroid_matrix, 
                             assignment_matrix) { 
  #' Updates the cluster centroid matrix according to 
  #' the average of the assigned points. 
  #' 
  #' Parameters: 
  #'   error_matrix: matrix
  #'     Matrix of dimensions N by the number of 
  #'     latent variables. 
  #'     
  #'   centroid_matrix: matrix 
  #'     Matrix of dimensions K by the number of 
  #'     latent variables. 
  #'     
  #'   assignment_matrix: matrix
  #'     Matrix of dimensions N by K. 
  #'     
  #' Return: 
  #'   Updated centroid matrix.
  
  # Iterate through the clusters 
  for (cluster in 1:ncol(assignment_matrix)) { 
    
    # Compute the error points associated with each cluster 
    cluster_error_points <- diag(assignment_matrix[, cluster]) %*% error_matrix
    
    # Compute the number of error points per cluster 
    total_points <- sum(assignment_matrix[, cluster])
    
    # Compute centroid distance
    centroid_matrix[cluster, ] <- colSums(cluster_error_points) / total_points
  }
  
  return(centroid_matrix)
  
}

# End of Update Cluster Centroids Function # 


#### Cluster Assignment #### 

# Compute Euclidean Distance Function # 

compute_euc_dis <- function(error_vector, 
                            centroid_vector) { 
  #' Computes the Euclidean distance of two vectors. 
  #' 
  #' Parameters: 
  #'   error_vector: vector 
  #'     A vector of the error of a given data point. 
  #'  
  #'   centroid_vector: vector
  #'     A vector of the cluster centroid.
  #'
  #' Return: 
  #'   A vector of dimensions 1 by K. 
  
  euc_dis <- sqrt(sum((error_vector - centroid_vector)^2))
  
  return(euc_dis)
  
}

# End of Compute Euclidean Distance Function # 

# Assign Clusters Function # 

assign_cluster <- function(error_matrix, 
                           centroid_matrix) { 
  #' Assign data points to their respective cluster according to 
  #' minimum Euclidean distance. 
  #' 
  #' Parameters: 
  #'   error_matrix: matrix 
  #'     A matrix of errors of dimensions N by the 
  #'     number of latent variables. 
  #'    
  #'   centroid_matrix: matrix 
  #'     A matrix of cluster centroids of dimensions K 
  #'     by Y. 
  #'    
  #' Return:
  #'   The assignment matrix of dimensions N by K. 

  # Initialize the assignment matrix. 
  assignment_matrix <- matrix(data = 0, 
                              nrow = nrow(x = error_matrix), 
                              ncol = nrow(x = centroid_matrix))
  
  # Iterate through the rows of the error matrix. 
  for (row in 1:nrow(error_matrix)) { 
    
    # Initialize Euclidean distance vector
    euc_dis_vector <- matrix(data = 0, 
                             nrow = 1,
                             ncol = nrow(x = centroid_matrix))
    
    # Iterate through the rows of the centroid matrix.
    for (col in 1:nrow(centroid_matrix)) { 
      # Compute the Euclidean distance.
      euc_dis_vector[1, col] <- compute_euc_dis(
        error_vector = error_matrix[row, ],
        centroid_vector = centroid_matrix[col, ]
        )
    }
    
    # Determine the minimum Euclidean distance for the data point. 
    assignment_col <- which.min(euc_dis_vector)
    
    # Assign the data point to its respective cluster.
    assignment_matrix[row, assignment_col] <- 1
  }
  
  return(assignment_matrix)
}

# End of Assign Clusters Function # 

#### Objective Function #### 

# Compute Objective Function # 

compute_obj_func <- function(error_matrix, 
                             assignment_matrix, 
                             centroid_matrix) {
  #' Computes the objective function. 
  #' 
  #' Parameters: 
  #'   error_matrix: matrix 
  #'     A matrix of dimensions N by Y of error estimates.
  #'     
  #'   assignment_matrix: matrix 
  #'     A matrix of dimensions N by K of binary cluster 
  #'     assignment. 
  #'     
  #'   centroid_matrix: matrix
  #'     A matrix of dimensions K by Y of cluster centroid
  #'     coordinates. 
  #'     
  #' Return: 
  #'   Scalar value of the model.
  
  # Initialize assignment-centroid matrix 
  # assignment_centroid_matrix <- assignment_matrix %*% centroid_matrix
  
  # Compute objective scalar value
  #obj_scalar <- sum(
   #x = (error_matrix - assignment_centroid_matrix)^2
    #)
  
  obj_scalar <- sum((error_matrix - (assignment_matrix %*% centroid_matrix))^2)
  
  return(obj_scalar) 
} 

# End of Compute Objective Function # 

#### Iteration Difference #### 

# Compute Iteration Difference Function # 

compute_iter_diff <- function(prev_obj_func,
                              curr_obj_func) { 
  #' Computes the difference between the current 
  #' and previous objective function. 
  #' 
  #' Parameters: 
  #'   prev_obj_func: scalar
  #'     The objective function scalar at iteration s-1.
  #'     
  #'   curr_obj_func: scalar
  #'     The objective function scalar at iteration s.
  #'   
  #' Return: 
  #'   A Boolean value.
  
  # Compares the difference of the objective function between iterations
  x <- prev_obj_func - curr_obj_func
  
  if (is.nan(curr_obj_func)) { 
    return(TRUE)
  } else if  (x < 0.00001) {
    return(FALSE)
  } else { 
    return(TRUE)  
  }
}
# End of Compute Iteration Difference Function # 


#### Intercept Estimation #### 

estimate_intercept <- function(Y, 
                               estimate, 
                               cluster_idx) { 
  #' Estimates the intercept of the generated data. 
  #' 
  #' Parameters: 
  #'   Y: matrix 
  #'     A matrix of dependent vectors.
  #'
  #'   estimate: matrix 
  #'     A matrix of forward propagation estimates. 
  #'     
  #'   cluster_idx: list 
  #'     A list of cluster indices. 
  #'     
  #'   cluster: int 
  #'    A value representing the cluster column.
  
  output <- colMeans(Y[cluster_idx, ]) - colMeans(estimate)
  
  return(output)
}

#### Main ####

# Main Function # 

main <- function(X_per_lv, 
                 HN_per_lv,
                 X,
                 Y,
                 k_clusters,
                 learning_rate) { 
  
  #' A main function that computes the kNN-ERA according to the 
  #' other functions in the package. 

  num_Y <- ncol(Y)
  input_layer_neurons <- ncol(X) 
  hidden_layer_neurons <- sum(HN_per_lv)
  lv_neurons <- length(HN_per_lv)
  output_neurons <- ncol(Y)
  
  #### Initialize general model weights #### 
  
  # Initialize weights 
  initial_W <- initialize_model_weights(layer_input = X_per_lv,
                                        layer_output = HN_per_lv,
                                        num_Y = num_Y)
  
  # Weight matrices
  wh <- initial_W$Z_0[[1]]
  wl <- initial_W$L_0[[1]]
  wout <- initial_W$A_0
  
  # Non-zero index 
  wh_non_zero <- initial_W$Z_0[[2]]
  wl_non_zero <- initial_W$L_0[[2]]
  
  # Storage 
  general_weights_storage <- list(wout, wl, wh)
  
  cluster_weights_storage <- list()
  
  for (i in 1:k_clusters) {
    cluster_weights_storage[[i]] <- list(wout, wl, wh)
  }
  
  #### Initialize model biases #### 
  
  # Initialize model biases 
  bias_hidden <- initialize_bias(X = X, 
                                 neuron_num = hidden_layer_neurons)
  
  bias_latent <- initialize_bias(X = X,
                                 neuron_num = lv_neurons)
  
  bias_out <- initialize_bias(X = X, 
                              neuron_num = output_neurons)
  
  # Storage 
  general_bias_list <- list(bias_out, bias_latent, bias_hidden)
  bias_list <- list(bias_out, bias_latent, bias_hidden)
  
  #### Forward propagation and activation #### 
  
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
  
  # Storage 
  general_forward_prop <- list(hidden_layer_output, latent_layer_output, output_layer_output)
  forward_prop_storage <- list()
  
  for (i in 1:k_clusters) { 
    forward_prop_storage[[i]] <- list(hidden_layer_output, 
                                      latent_layer_output,
                                      output_layer_output)
    }
  
  #### Error and centroid matrix #### 
  
  # Compute error 
  E <- compute_error(y_matrix = Y, 
                     y_predicted_matrix = output_layer_output)
  
  storage <- E
  mse_storage <- E
  relative_bias_storage <- E
  general_error <- E
  
  # Initialize centroid matrix
  centroid_matrix <- initialize_centroids(k_clusters = k_clusters, 
                                          error_matrix = E)
  
  #### Cluster assignment #### 
  
  # Initialize assignment 
  assignment_matrix <- assign_cluster(error_matrix = E,
                                      centroid_matrix = centroid_matrix)

  ##### Convergence ##### 
  
  prev_cw_error <- 1e13
  curr_cw_error <- compute_obj_func(general_error,
                                    assignment_matrix,
                                    centroid_matrix)
  
  while (compute_iter_diff(prev_cw_error, curr_cw_error)) { 

    prev_cw_error <- curr_cw_error
    
    #### General feed-forward #### 
    
    # Hidden layer # 
    general_hidden_layer_input <- forward_prop(layer = "hidden",
                                               input_matrix = X,
                                               weight_matrix = general_weights_storage[[3]],
                                               layer_bias = general_bias_list[[3]])
    
    general_forward_prop[[1]] <- activation(layer_matrix = general_hidden_layer_input,
                                            activation_function = "sigmoid")
    
    # Latent layer # 
    general_latent_layer_input <- forward_prop(layer = "latent", 
                                               input_matrix = general_forward_prop[[1]], 
                                               weight_matrix = general_weights_storage[[2]], 
                                               layer_bias = general_bias_list[[2]])
    
    general_forward_prop[[2]]  <- activation(layer_matrix = general_latent_layer_input,
                                             activation_function = "linear")
    
    # Output layer # 
    general_output_layer_input <- forward_prop(layer = "output",
                                               input_matrix = general_forward_prop[[2]], 
                                               weight_matrix = general_weights_storage[[1]],
                                               layer_bias = general_bias_list[[1]])
    
    general_forward_prop[[3]]  <- activation(layer_matrix = general_output_layer_input,
                                             activation_function = "linear")
    
    #### General error #### 
    
    # Compute error 
    general_error <- compute_error(y_matrix = Y,
                                   y_predicted_matrix = general_forward_prop[[3]])
    
    #### Update centroid matrix #### 
    
    centroid_matrix <- update_centroids(error_matrix = general_error,
                                        centroid_matrix = centroid_matrix, 
                                        assignment_matrix = assignment_matrix)

    
    #### Update cluster assignment #### 
    
    assignment_matrix <- assign_cluster(error_matrix = general_error,
                                        centroid_matrix = centroid_matrix)
    
    # Initialize intercept estimate storage
    
    intercept_estimates <- list()
    
    #### Cluster-wise NN-ERA #### 
    
    for (col in 1:ncol(assignment_matrix)) { 
      
      #### Cluster point indices #### 
      
      cluster_idx <- which(assignment_matrix[, col] == 1)
      centroid_points <- X[cluster_idx, ]
      
      #### Cluster-wise feed forward ####
      
      #Hidden layer #
      hidden_layer_input <- forward_prop(layer = "hidden",
                                         input_matrix = centroid_points,
                                         weight_matrix = cluster_weights_storage[[col]][[3]],
                                         layer_bias = bias_list[[3]][cluster_idx, ])
      
      forward_prop_storage[[col]][[1]] <- activation(layer_matrix = hidden_layer_input,
                                                     activation_function = "sigmoid")
      
      # Latent layer # 
      latent_layer_input <- forward_prop(layer = "latent", 
                                         input_matrix = forward_prop_storage[[col]][[1]], 
                                         weight_matrix = cluster_weights_storage[[col]][[2]], 
                                         layer_bias = bias_list[[2]][cluster_idx, ])
      
      forward_prop_storage[[col]][[2]]  <- activation(layer_matrix = latent_layer_input,
                                                      activation_function = "linear")
      
      # Output layer # 
      output_layer_input <- forward_prop(layer = "output",
                                         input_matrix = forward_prop_storage[[col]][[2]], 
                                         weight_matrix = cluster_weights_storage[[col]][[1]],
                                         layer_bias = bias_list[[1]][cluster_idx,])
      
      forward_prop_storage[[col]][[3]]  <- activation(layer_matrix = output_layer_input,
                                                      activation_function = "linear")
      
      # Compute intercept estimate 
      
      intercept_estimates[[col]] <- estimate_intercept(Y = Y,
                                                       estimate = forward_prop_storage[[col]][[3]],
                                                       cluster_idx = cluster_idx)
      
      #### Storage ####
      
      # add the estimated intercept to the estimates 
      
      total_est <- matrix(forward_prop_storage[[col]][[3]], ncol = 2) + matrix(rep(intercept_estimates[[col]], length(cluster_idx)), length(cluster_idx), byrow = TRUE)
      
      storage[cluster_idx, ] <- compute_error(y_matrix = matrix(Y[cluster_idx,], ncol = 2),
                                              y_predicted_matrix = total_est)
      
      mse_storage[cluster_idx, ] <- compute_mse(y_matrix = matrix(Y[cluster_idx, ], ncol = 2), 
                                                y_predicted_matrix = total_est)
      
      relative_bias_storage[cluster_idx, ] <- compute_relative_bias(y_matrix = matrix(Y[cluster_idx,], ncol = 2), 
                                                                    y_predicted_matrix = total_est)
      #### Cluster-wise back propagation #### 
      
      # Layer activation derivatives # 
      
      slope_output_layer <- activation_derivative(layer_matrix = forward_prop_storage[[col]][[3]],
                                                  activation_function = "linear")
      
      slope_latent_layer <- activation_derivative(layer_matrix = forward_prop_storage[[col]][[2]], 
                                                  activation_function = "linear")
      
      slope_hidden_layer <- activation_derivative(layer_matrix = forward_prop_storage[[col]][[1]],
                                                  activation_function = "sigmoid")
      
      # Layer back propagation # 
      
      d_output <- back_prop(layer = "output",
                            derivative_input_matrix = storage[cluster_idx, ],
                            layer_derivative_matrix = slope_output_layer) 
      
      d_latent <- back_prop(layer = "latent",
                            derivative_input_matrix = d_output,
                            layer_derivative_matrix = slope_latent_layer,
                            weight_matrix = cluster_weights_storage[[col]][[1]])
      
      d_hidden <- back_prop(layer = "hidden",
                            derivative_input_matrix = d_latent,
                            layer_derivative_matrix = slope_hidden_layer,
                            weight_matrix = cluster_weights_storage[[col]][[2]])
      
      #### Cluster-wise weight updates ####
      
      cluster_weights_storage[[col]][[1]] <- update_weights(layer = "output", 
                                                            weight_matrix = cluster_weights_storage[[col]][[1]], 
                                                            input_matrix = forward_prop_storage[[col]][[2]],
                                                            derivative_input_matrix = d_output,
                                                            learning_rate = learning_rate) 
      
      cluster_weights_storage[[col]][[2]] <- update_weights(layer = "latent", 
                                                            weight_matrix = cluster_weights_storage[[col]][[2]], 
                                                            input_matrix = forward_prop_storage[[col]][[1]],
                                                            derivative_input_matrix = d_latent,
                                                            learning_rate = learning_rate,
                                                            non_zero_weight_indices = wl_non_zero) 
      
      cluster_weights_storage[[col]][[3]] <- update_weights(layer = "hidden", 
                                                            weight_matrix = cluster_weights_storage[[col]][[3]], 
                                                            input_matrix = centroid_points,
                                                            derivative_input_matrix = d_hidden,
                                                            learning_rate = learning_rate,
                                                            non_zero_weight_indices = wh_non_zero)
      
      #### Cluster-wise bias updates #### 
      
      bias_list[[1]][cluster_idx, ] <- update_bias(layer = "output",
                                                   bias_matrix = bias_list[[1]][cluster_idx, ],
                                                   layer_derivative_matrix = d_output,
                                                   learning_rate = learning_rate)
      
      bias_list[[2]][cluster_idx, ] <- update_bias(layer = "latent",
                                                   bias_matrix = bias_list[[2]][cluster_idx, ],
                                                   layer_derivative_matrix = d_latent,
                                                   learning_rate = learning_rate)
      
      bias_list[[3]][cluster_idx, ] <- update_bias(layer = "hidden",
                                                   bias_matrix = bias_list[[3]][cluster_idx, ],
                                                   layer_derivative_matrix = d_hidden,
                                                   learning_rate = learning_rate,
                                                   hdn_node_per_lv = HN_per_lv)
    }

    
    #### General back-propagation ####
    
    general_slope_output_layer <- activation_derivative(layer_matrix = general_forward_prop[[3]],
                                                        activation_function = "linear")
    
    general_slope_latent_layer <- activation_derivative(layer_matrix = general_forward_prop[[2]],
                                                        activation_function = "linear")
    
    general_slope_hidden_layer <- activation_derivative(layer_matrix = general_forward_prop[[1]],
                                                        activation_function = "sigmoid")
    
    general_d_output <- back_prop(layer = "output",
                                  derivative_input_matrix = general_error,
                                  layer_derivative_matrix = general_slope_output_layer)
    
    general_d_latent <- back_prop(layer = "latent",
                                  derivative_input_matrix = general_d_output,
                                  layer_derivative_matrix = general_slope_latent_layer,
                                  weight_matrix = general_weights_storage[[1]])
    
    general_d_hidden <- back_prop(layer = "hidden",
                                  derivative_input_matrix = general_d_latent,
                                  layer_derivative_matrix = general_slope_hidden_layer,
                                  weight_matrix = general_weights_storage[[2]])
    
    #### General weight update ####
    
    general_weights_storage[[1]] <- update_weights(layer = "output",
                                                   weight_matrix = general_weights_storage[[1]],
                                                   input_matrix = general_forward_prop[[2]],
                                                   derivative_input_matrix = general_d_output,
                                                   learning_rate = learning_rate)
    
    general_weights_storage[[2]] <- update_weights(layer = "latent",
                                                   weight_matrix = general_weights_storage[[2]],
                                                   input_matrix = general_forward_prop[[1]],
                                                   derivative_input_matrix = general_d_latent,
                                                   learning_rate = learning_rate,
                                                   non_zero_weight_indices = wl_non_zero)
    
    general_weights_storage[[3]] <- update_weights(layer = "hidden",
                                                   weight_matrix = general_weights_storage[[3]],
                                                   input_matrix = X,
                                                   derivative_input_matrix = general_d_hidden,
                                                   learning_rate = learning_rate,
                                                   non_zero_weight_indices = wh_non_zero)
    
    #### General bias update ####
    
    general_bias_list[[1]] <- update_bias(layer = "output",
                                          bias_matrix = general_bias_list[[1]],
                                          layer_derivative_matrix = general_d_output,
                                          learning_rate = learning_rate)
    
    general_bias_list[[2]] <- update_bias(layer = "latent",
                                          bias_matrix = general_bias_list[[2]],
                                          layer_derivative_matrix = general_d_latent,
                                          learning_rate = learning_rate)
    
    general_bias_list[[3]] <- update_bias(layer = "hidden",
                                          bias_matrix = general_bias_list[[3]],
                                          layer_derivative_matrix = general_d_hidden,
                                          learning_rate = learning_rate,
                                          hdn_node_per_lv = HN_per_lv)
    
    curr_cw_error <- sum(storage)
    
  }

  mse <- sum(mse_storage)/nrow(mse_storage)
  
  relative_bias <- sum(relative_bias_storage)/nrow(relative_bias_storage)
  
  return (list(assignment_matrix, 
               storage, 
               centroid_matrix, 
               cluster_weights_storage,
               intercept_estimates,
               forward_prop_storage,
               mse,
               general_error,
               relative_bias,
               mse_storage))
  
  
  }

# End of Main Function # 

