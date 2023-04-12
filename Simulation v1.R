# Simulation Data and Model Comparison
library("R.utils")
library("writexl")
source("kNN-ERA Functions.R")
source("Step-Wise Functions.R")

set.seed(1234)

# Data formats: 

# Compare cluster identification between k-means and kNN-ERA 

# Compare cluster params between NN-ERA and kNN-ERA 

#' Simulation Data and Model Comparison 
#' 
#' Data formats: 
#'   Number of clusters: 2
#'   Sample size: 100, 300, 1000
#'   Cluster size: 50%, 75%
#'   Variance in the error terms: High or low
#'   
#'   Total of twelve conditions 

#### Specifications of the true model ####

K_CLUSTER <- 2

# Intercept Matrix 

c1_int <- matrix(data = c(10, 10), 
                 nrow = 1, 
                 ncol = 2)

c2_int <- matrix(data = c(-5, -5), 
                 nrow = 1,
                 ncol = 2)

int_list <- list(c1_int, c2_int)

# W Matrix 
c1_w <- matrix(data = c(0.8, 0.8, 0, 0, 
                        0, 0, 0.8, 0.8),
               nrow = 4,
               ncol = 2)

c2_w <- matrix(data = c(0.4, 0.4, 0, 0,
                        0, 0, 0.4, 0.4), 
               nrow = 4,
               ncol = 2)

w_list <- list(c1_w, c2_w)

# A Matrix
c1_a <- matrix(data = c(0, 0.7,
                        0.7, 0.3),
               nrow = 2,
               ncol = 2)

c2_a <- matrix(data = c(0, 0.4,
                        0.4, 0.1),
               nrow = 2,
               ncol = 2)

a_list <- list(c1_a, c2_a)

# X Variance (Low)

c1_x_sig_low <- matrix(data = c(1, 0.8, 0.3, 0.3,
                                0.8, 1, 0.3, 0.3,
                                0.3, 0.3, 1, 0.8, 
                                0.3, 0.3, 0.8, 1),
                       nrow = 4, 
                       ncol = 4)

c2_x_sig_low <- matrix(data = c(1, 0.8, 0.3, 0.3,
                                0.8, 1, 0.3, 0.3,
                                0.3, 0.3, 1, 0.8, 
                                0.3, 0.3, 0.8, 1),
                       nrow = 4, 
                       ncol = 4)

x_sig_list_low <- list(c1_x_sig_low, c2_x_sig_low)

# X Variance (High) 

c1_x_sig_high <- matrix(data = c(1, 0.8, 0.6, 0.6,
                                 0.8, 1, 0.6, 0.6,
                                 0.6, 0.6, 1, 0.8, 
                                 0.6, 0.6, 0.8, 1),
                        nrow = 4, 
                        ncol = 4)

c2_x_sig_high <- matrix(data = c(1, 0.8, 0.6, 0.6,
                                 0.8, 1, 0.6, 0.6,
                                 0.6, 0.6, 1, 0.8, 
                                 0.6, 0.6, 0.8, 1),
                        nrow = 4, 
                        ncol = 4)

x_sig_list_high <- list(c1_x_sig_high, c2_x_sig_high)

# Y Variance 

c1_y_sig <- matrix(data = c(1.2, 0.1,
                            0.1, 0.8), 
                   nrow = 2, 
                   ncol = 2)

c2_y_sig <- matrix(data = c(1.2, 0.3,
                            0.3, 0.8), 
                   nrow = 2, 
                   ncol = 2)

y_sig_list <- list(c1_y_sig, c2_y_sig)



#### 100/50%/low ####

CLUSTER_SIZE_LIST <- list(50, 50)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate <- matrix(data = 0, 
                                  nrow = 500, 
                                  ncol = 1)
mse <- matrix(data = 0, 
              nrow = 500, 
              ncol = 1)
relative_bias <- matrix(data = 0, 
                        nrow = 500, 
                        ncol = 1)

# Step-wise storage
sw_avg_classification_rate <- matrix(data = 0, 
                                     nrow = 500, 
                                     ncol = 1)
sw_mse <- matrix(data = 0, 
                 nrow = 500, 
                 ncol = 1)
sw_relative_bias <- matrix(data = 0, 
                           nrow = 500, 
                           ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate[i] <- compute_classification_rate(output = output[[1]],
                                                            proportion = "50")
  
  mse[i] <- output[[7]]
  
  relative_bias[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias[i] <- sum(step_wise_output[[2]][[6]],
                             step_wise_output[[2]][[12]])
  
  print(i)
  
}

df <- data.frame(unifed_classification_rate = avg_classification_rate,
                 unified_mse = mse,
                 unified_relative_bias = relative_bias,
                 sw_classification_rate = sw_avg_classification_rate,
                 sw_mse = sw_mse,
                 sw_relative_bias = sw_relative_bias)

for (col in 1:ncol(df)) { 
  print(colnames(df[col]))
  print(mean(df[, col]))
  print(sd(df[, col]))
  }

#### 100/50%/high #### 

CLUSTER_SIZE_LIST <- list(50, 50)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_2 <- matrix(data = 0, 
                                  nrow = 500, 
                                  ncol = 1)
mse_2 <- matrix(data = 0, 
              nrow = 500, 
              ncol = 1)
relative_bias_2 <- matrix(data = 0, 
                        nrow = 500, 
                        ncol = 1)

# Step-wise storage
sw_avg_classification_rate_2 <- matrix(data = 0, 
                                     nrow = 500, 
                                     ncol = 1)
sw_mse_2 <- matrix(data = 0, 
                 nrow = 500, 
                 ncol = 1)
sw_relative_bias_2 <- matrix(data = 0, 
                           nrow = 500, 
                           ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_2[i] <- compute_classification_rate(output = output[[1]],
                                                            proportion = "50")
  
  mse_2[i] <- output[[7]]
  
  relative_bias_2[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_2[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse_2[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_2[i] <- sum(step_wise_output[[2]][[6]],
                             step_wise_output[[2]][[12]])
  
  print(i)
  
}

df2 <- data.frame(unifed_classification_rate = avg_classification_rate_2,
                 unified_mse = mse_2,
                 unified_relative_bias = relative_bias_2,
                 sw_classification_rate = sw_avg_classification_rate_2,
                 sw_mse = sw_mse_2,
                 sw_relative_bias = sw_relative_bias_2)

for (col in 1:ncol(df)) { 
  print(colnames(df2[col]))
  print(mean(df2[, col]))
  print(sd(df2[, col]))
}


#### 100/75%/low #### 

CLUSTER_SIZE_LIST <- list(75, 25)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_3 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_3 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_3 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_3 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)

sw_mse_3 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_3 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_3[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "75")
  
  mse_3[i] <- output[[7]]
  
  relative_bias_3[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_3[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_3[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_3[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df3 <- data.frame(unifed_classification_rate = avg_classification_rate_3,
                  unified_mse = mse_3,
                  unified_relative_bias = relative_bias_3,
                  sw_classification_rate = sw_avg_classification_rate_3,
                  sw_mse = sw_mse_3,
                  sw_relative_bias = sw_relative_bias_3)

for (col in 1:ncol(df)) { 
  print(colnames(df3[col]))
  print(round(mean(df3[, col]), 3))
  print(round(sd(df3[, col]), 3))
        
}

#### 100/75%/high #### 

CLUSTER_SIZE_LIST <- list(75, 25)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_4 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_4 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_4 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_4 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)

sw_mse_4 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_4 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)


for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_4[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "75")
  
  mse_4[i] <- output[[7]]
  
  relative_bias_4[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_4[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_4[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_4[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df4 <- data.frame(unifed_classification_rate = avg_classification_rate_4,
                  unified_mse = mse_4,
                  unified_relative_bias = relative_bias_4,
                  sw_classification_rate = sw_avg_classification_rate_4,
                  sw_mse = sw_mse_4,
                  sw_relative_bias = sw_relative_bias_4)

for (col in 1:ncol(df)) { 
  print(colnames(df4[col]))
  print(round(mean(df4[, col]), 3))
  print(round(sd(df4[, col]), 3))
}

#### 300/50%/low ####

CLUSTER_SIZE_LIST <- list(150, 150)
CLUSTER_MEAN_LIST <- list(10, -10)


# Unified storage 
avg_classification_rate_5 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_5 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_5 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_5 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_5 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_5 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_5[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "50")
  
  mse_5[i] <- output[[7]]
  
  relative_bias_5[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_5[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse_5[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_5[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df5 <- data.frame(unifed_classification_rate = avg_classification_rate_5,
                  unified_mse = mse_5,
                  unified_relative_bias = relative_bias_5,
                  sw_classification_rate = sw_avg_classification_rate_5,
                  sw_mse = sw_mse_5,
                  sw_relative_bias = sw_relative_bias_5)

for (col in 1:ncol(df)) { 
  print(colnames(df[col]))
  print(round(mean(df5[, col]), 3))
  print(round(sd(df5[, col]), 3))
}

#### 300/50%/high ####

CLUSTER_SIZE_LIST <- list(150, 150)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_6 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_6 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_6 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_6 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_6 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_6 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_6[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "50")
  
  mse_6[i] <- output[[7]]
  
  relative_bias_6[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_6[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse_6[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_6[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df6 <- data.frame(unifed_classification_rate = avg_classification_rate_6,
                  unified_mse = mse_6,
                  unified_relative_bias = relative_bias_6,
                  sw_classification_rate = sw_avg_classification_rate_6,
                  sw_mse = sw_mse_6,
                  sw_relative_bias = sw_relative_bias_6)


for (col in 1:ncol(df)) { 
  print(colnames(df[col]))
  print(round(mean(df6[, col]), 3))
  print(round(sd(df6[, col]), 3))
}

#### 300/75%/low ####

CLUSTER_SIZE_LIST <- list(225, 75)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_7 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_7 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_7 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_7 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_7 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_7 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_7[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "75")
  
  mse_7[i] <- output[[7]]
  
  relative_bias_7[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_7[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_7[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_7[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df7 <- data.frame(unifed_classification_rate = avg_classification_rate_7,
                  unified_mse = mse_7,
                  unified_relative_bias = relative_bias_7,
                  sw_classification_rate = sw_avg_classification_rate_7,
                  sw_mse = sw_mse_7,
                  sw_relative_bias = sw_relative_bias_7)

#### 300/75%/high ####

CLUSTER_SIZE_LIST <- list(225, 75)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_8 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_8 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_8 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_8 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_8 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_8 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_8[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "75")
  
  mse_8[i] <- output[[7]]
  
  relative_bias_8[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_8[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_8[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_8[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df8 <- data.frame(unifed_classification_rate = avg_classification_rate_8,
                  unified_mse = mse_8,
                  unified_relative_bias = relative_bias_8,
                  sw_classification_rate = sw_avg_classification_rate_8,
                  sw_mse = sw_mse_8,
                  sw_relative_bias = sw_relative_bias_8)

sheets <- list("100 50 low" = df, 
               "100 50 high" = df2, 
               "100 75 low" = df3,
               "100 75 high" = df4,
               "300 50 low" = df5,
               "300 50 high" = df6,
               "300 75 low" = df7,
               "300 75 high" = df8)

write_xlsx(sheets, path = "output.xlsx")

#### 1000/50%/low ####

CLUSTER_SIZE_LIST <- list(500, 500)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_9 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_9 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_9 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_9 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_9 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_9 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_9[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "50")
  
  mse_9[i] <- output[[7]]
  
  relative_bias_9[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_9[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse_9[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_9[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}


df9 <- data.frame(unifed_classification_rate = avg_classification_rate_9,
                  unified_mse = mse_9,
                  unified_relative_bias = relative_bias_9,
                  sw_classification_rate = sw_avg_classification_rate_9,
                  sw_mse = sw_mse_9,
                  sw_relative_bias = sw_relative_bias_9)

for (col in 1:ncol(df)) { 
  print(colnames(df[col]))
  print(round(mean(df9[, col]), 3))
  print(round(sd(df9[, col]), 3))
}

#### 1000/50%/high ####

CLUSTER_SIZE_LIST <- list(500, 500)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_10 <- matrix(data = 0, 
                                    nrow = 500, 
                                    ncol = 1)
mse_10 <- matrix(data = 0, 
                nrow = 500, 
                ncol = 1)
relative_bias_10 <- matrix(data = 0, 
                          nrow = 500, 
                          ncol = 1)

# Step-wise storage
sw_avg_classification_rate_10 <- matrix(data = 0, 
                                       nrow = 500, 
                                       ncol = 1)
sw_mse_10 <- matrix(data = 0, 
                   nrow = 500, 
                   ncol = 1)
sw_relative_bias_10 <- matrix(data = 0, 
                             nrow = 500, 
                             ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_10[i] <- compute_classification_rate(output = output[[1]],
                                                              proportion = "50")
  
  mse_10[i] <- output[[7]]
  
  relative_bias_10[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_10[i] <- compute_classification_rate(sw_class_mat, "50")
  
  sw_mse_10[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_10[i] <- sum(step_wise_output[[2]][[6]],
                               step_wise_output[[2]][[12]])
  
  print(i)
  
}

df10 <- data.frame(unifed_classification_rate = avg_classification_rate_10,
                  unified_mse = mse_10,
                  unified_relative_bias = relative_bias_10,
                  sw_classification_rate = sw_avg_classification_rate_10,
                  sw_mse = sw_mse_10,
                  sw_relative_bias = sw_relative_bias_10)

#### 1000/75%/low ####

CLUSTER_SIZE_LIST <- list(750, 250)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_11 <- matrix(data = 0, 
                                     nrow = 500, 
                                     ncol = 1)

mse_11 <- matrix(data = 0, 
                 nrow = 500, 
                 ncol = 1)
relative_bias_11 <- matrix(data = 0, 
                           nrow = 500, 
                           ncol = 1)

# Step-wise storage
sw_avg_classification_rate_11 <- matrix(data = 0, 
                                        nrow = 500, 
                                        ncol = 1)

sw_mse_11 <- matrix(data = 0, 
                    nrow = 500, 
                    ncol = 1)
sw_relative_bias_11 <- matrix(data = 0, 
                              nrow = 500, 
                              ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_low, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 10,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_11[i] <- compute_classification_rate(output = output[[1]],
                                                               proportion = "75")
  
  mse_11[i] <- output[[7]]
  
  relative_bias_11[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_11[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_11[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_11[i] <- sum(step_wise_output[[2]][[6]],
                                step_wise_output[[2]][[12]])
  
  print(i)
  
}

df11 <- data.frame(unifed_classification_rate = avg_classification_rate_11,
                   unified_mse = mse_11,
                   unified_relative_bias = relative_bias_11,
                   sw_classification_rate = sw_avg_classification_rate_11,
                   sw_mse = sw_mse_11,
                   sw_relative_bias = sw_relative_bias_11)


#### 1000/75%/high ####

CLUSTER_SIZE_LIST <- list(750, 250)
CLUSTER_MEAN_LIST <- list(10, -10)

# Unified storage 
avg_classification_rate_12 <- matrix(data = 0, 
                                     nrow = 500, 
                                     ncol = 1)
mse_12 <- matrix(data = 0, 
                 nrow = 500, 
                 ncol = 1)
relative_bias_12 <- matrix(data = 0, 
                           nrow = 500, 
                           ncol = 1)

# Step-wise storage
sw_avg_classification_rate_12 <- matrix(data = 0, 
                                        nrow = 500, 
                                        ncol = 1)
sw_mse_12 <- matrix(data = 0, 
                    nrow = 500, 
                    ncol = 1)
sw_relative_bias_12 <- matrix(data = 0, 
                              nrow = 500, 
                              ncol = 1)

for (i in 1:500) { 
  
  data <- generate_data(int_mat = int_list, 
                        w_mat = w_list, 
                        a_mat = a_list, 
                        x_sig_mat = x_sig_list_high, 
                        y_sig_mat = y_sig_list, 
                        k_clusters = K_CLUSTER,
                        cluster_size = CLUSTER_SIZE_LIST,
                        cluster_mean = CLUSTER_MEAN_LIST)
  
  # kNN-ERA Parameters 
  
  dv_idx <- c(1, 2)
  
  y <- as.matrix(data[, dv_idx])
  x <- as.matrix(data[, -dv_idx])
  
  x_per_lv <- c(2, 2)
  hdn_per_lv <- c(4, 4)
  learning_rate <- 0.0001
  
  num_Y <- ncol(y)
  input_layer_neurons <- ncol(x) 
  hdn_layer_neurons <- sum(hdn_per_lv)
  lv_neurons <- length(hdn_per_lv)
  output_neurons <- ncol(y)
  
  tryCatch({
    withTimeout(output <- main(x_per_lv,hdn_per_lv, x, y,
                               K_CLUSTER, learning_rate),
                timeout = 15,
                onTimeout = "silent")
  },
  error = function(e) e)
  
  tryCatch({
    withTimeout(step_wise_output <- compute_step_wise(k = K_CLUSTER,
                                                      x = x, 
                                                      y = y, 
                                                      x_per_lv = x_per_lv, 
                                                      hdn_per_lv = hdn_per_lv,
                                                      hdn_layer_neurons = hdn_layer_neurons,
                                                      lv_neurons = lv_neurons, 
                                                      output_neurons = output_neurons,
                                                      iterations = 2000,
                                                      num_y = num_Y),
                timeout = 10,
                onTimeout = "silent")
  }, 
  error = function(e) e)
  
  # Unified metrics 
  avg_classification_rate_12[i] <- compute_classification_rate(output = output[[1]],
                                                               proportion = "75")
  
  mse_12[i] <- output[[7]]
  
  relative_bias_12[i] <- output[[9]]
  
  # Step-wise metrics
  sw_class_mat <- convert_assignment_vec(step_wise_output[[1]]$cluster)
  sw_avg_classification_rate_12[i] <- compute_classification_rate(sw_class_mat, "75")
  
  sw_mse_12[i] <- sum(step_wise_output[[2]][[4]], step_wise_output[[2]][[10]])
  
  sw_relative_bias_12[i] <- sum(step_wise_output[[2]][[6]],
                                step_wise_output[[2]][[12]])
  
  print(i)
  
}

df12 <- data.frame(unifed_classification_rate = avg_classification_rate_12,
                   unified_mse = mse_12,
                   unified_relative_bias = relative_bias_12,
                   sw_classification_rate = sw_avg_classification_rate_12,
                   sw_mse = sw_mse_12,
                   sw_relative_bias = sw_relative_bias_12)

sheets <- list("100 50 low" = df,
               "100 50 high" = df2,
               "100 75 low" = df3,
               "100 75 high" = df4,
               "300 50 low" = df5,
               "300 50 high" = df6,
               "300 75 low" = df7,
               "300 75 high" = df8,
               "1000 50 low" = df9,
               "1000 50 high" = df10,
               "1000 75 low" = df11,
               "1000 75 high" = df12)

write_xlsx(sheets, path = "output.xlsx")

