library(ggplot2)
library(caret)  # For k-fold cross-validation

# Datos de entrada 
datos <- data.frame(
  VG = c(70,50,70,70,50,50,70,60,60,50,50,70,60,50,60,50,70,70,70,50),
  IP = c(10,20,20,20,20,20,10,15,15,10,10,20,15,10,15,20,10,10,20,10),
  F = c(40,40,40,10,10,40,10,25,25,40,10,10,25,40,25,10,10,40,40,10),
  MRR = c(10.4381,10.168,28.4994,28.3038,9.7289,9.8883,13.7187,17.0721,
          17.6203,5.2211,4.374,29.1404,16.9155,5.2241,18.6677,9.7334,
          14.0191,12.9771,26.0022,4.2508)
)

# Función de preparación de datos con normalización
prepare_data <- function(datos) {
  # Normalización min-max
  normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
  }
  
  # Crear objeto de datos normalizados
  datos_norm <- data.frame(
    VG = normalize(datos$VG),
    IP = normalize(datos$IP),
    F = normalize(datos$F),
    MRR = datos$MRR  # No normalizar la variable objetivo
  )
  
  return(datos_norm)
}

# Función de pertenencia gaussiana 
gaussian_membership <- function(x, centro, amplitud, sigma = 1) {
  # Añadir parámetro sigma para más flexibilidad
  exp(-((x - centro)^2) / (2 * (amplitud * sigma)^2))
}

# Función de inferencia fuzzy con más flexibilidad
fuzzy_inference <- function(entrada, parametros) {
  # Calcular membresías
  memb_vg <- gaussian_membership(entrada$VG, 
                                 parametros$VG$centro, 
                                 parametros$VG$amplitud)
  memb_ip <- gaussian_membership(entrada$IP, 
                                 parametros$IP$centro, 
                                 parametros$IP$amplitud)
  memb_f <- gaussian_membership(entrada$F, 
                                parametros$F$centro, 
                                parametros$F$amplitud)
  
  # Reglas fuzzy con pesos adaptativos
  peso_vg <- parametros$weights[1]
  peso_ip <- parametros$weights[2]
  peso_f <- parametros$weights[3]
  
  # Inferencia con más capas de complejidad
  activacion <- (
    memb_vg * peso_vg + 
      (1 - memb_ip) * peso_ip + 
      memb_f * peso_f
  ) / sum(parametros$weights)
  
  # Desnormalizar salida
  salida <- activacion * parametros$scale_factor * max(datos$MRR)
  return(salida)
}

# Función objetivo con regularización adaptativa
f_obj <- function(real, pred, params, lambda = 0.0001) {
  # Mean Absolute Percentage Error (MAPE)
  mape <- mean(abs((real - pred) / real))
  
  # Regularización adaptativa
  reg_term <- lambda * (
    sum(params$weights^2) + 
      params$VG$amplitud^2 +
      params$IP$amplitud^2 + 
      params$F$amplitud^2
  )
  
  return(mape + reg_term)
}

# Algoritmo genético avanzado con k-fold cross-validation
advanced_genetic_algorithm <- function(
    datos_norm, 
    k_folds = 5,
    pop_size = 400,
    num_generations = 20000,
    mutation_rate_start = 0.1,
    mutation_rate_end = 0.005,
    patience = 75
) {
  # Preparar folds para cross-validation
  set.seed(123)
  folds <- createFolds(datos_norm$MRR, k = k_folds)
  
  # Matrices para almacenar resultados de cada fold
  all_mape_results <- numeric()
  
  # Función para ejecutar un fold
  run_fold <- function(fold_test_idx) {
    # Dividir datos
    datos_val <- datos_norm[fold_test_idx,]
    datos_train <- datos_norm[-fold_test_idx,]
    
    # Inicialización de población
    population <- matrix(
      data = runif(pop_size * 13, min = 0, max = 1), 
      nrow = pop_size
    )
    
    best_overall_error <- Inf
    generations_no_improvement <- 0
    
    for (gen in 1:num_generations) {
      # Evaluación de población
      errors <- apply(population, 1, function(cromosoma) {
        params <- list(
          VG = list(centro = cromosoma[1], amplitud = cromosoma[2] * 0.5),
          IP = list(centro = cromosoma[3], amplitud = cromosoma[4] * 0.5),
          F = list(centro = cromosoma[5], amplitud = cromosoma[6] * 0.5),
          weights = cromosoma[7:9],
          scale_factor = cromosoma[10]
        )
        
        pred_train <- sapply(1:nrow(datos_train), function(i) {
          fuzzy_inference(datos_train[i,], params)
        })
        
        return(f_obj(datos_train$MRR, pred_train, params))
      })
      
      # Selección de élite
      elite_size <- max(2, floor(0.15 * pop_size))
      elite_idx <- order(errors)[1:elite_size]
      new_pop <- population[elite_idx,]
      
      # Proceso de reproducción y mutación similar al anterior
      while (nrow(new_pop) < pop_size) {
        tournament_size <- 7
        tournament_idx <- sample(1:pop_size, tournament_size)
        winner_idx <- tournament_idx[which.min(errors[tournament_idx])]
        
        tournament_idx2 <- sample(1:pop_size, tournament_size)
        winner_idx2 <- tournament_idx2[which.min(errors[tournament_idx2])]
        
        # Crossover
        nc <- 2
        u <- runif(ncol(population))
        beta <- ifelse(u <= 0.5,
                       (2 * u)^(1/(nc + 1)),
                       (1/(2 * (1 - u)))^(1/(nc + 1)))
        
        child1 <- 0.5 * ((1 + beta) * population[winner_idx,] +
                           (1 - beta) * population[winner_idx2,])
        child2 <- 0.5 * ((1 - beta) * population[winner_idx,] +
                           (1 + beta) * population[winner_idx2,])
        
        new_pop <- rbind(new_pop, child1, child2)
      }
      
      # Recortar y mutar población
      if (nrow(new_pop) > pop_size) {
        new_pop <- new_pop[1:pop_size,]
      }
      
      # Mutación adaptativa
      mutation_rate <- mutation_rate_start -
        (gen/num_generations) * (mutation_rate_start - mutation_rate_end)
      
      for (i in 1:nrow(new_pop)) {
        if (runif(1) < mutation_rate) {
          gene <- sample(1:ncol(new_pop), 1)
          delta <- ifelse(runif(1) < 0.5,
                          (2 * runif(1))^(1/21) - 1,
                          1 - (2 * (1 - runif(1)))^(1/21))
          new_pop[i, gene] <- new_pop[i, gene] + delta * 0.1
          new_pop[i, gene] <- pmin(pmax(new_pop[i, gene], 0), 1)
        }
      }
      
      # Normalizar pesos
      new_pop[,7:9] <- t(apply(new_pop[,7:9], 1, function(x) x/sum(x)))
      
      population <- new_pop
      
      # Validación
      best_idx <- which.min(errors)
      best_params <- list(
        VG = list(centro = population[best_idx,1],
                  amplitud = population[best_idx,2] * 0.5),
        IP = list(centro = population[best_idx,3],
                  amplitud = population[best_idx,4] * 0.5),
        F = list(centro = population[best_idx,5],
                 amplitud = population[best_idx,6] * 0.5),
        weights = population[best_idx,7:9],
        scale_factor = population[best_idx,10]
      )
      
      pred_val <- sapply(1:nrow(datos_val), function(i) {
        fuzzy_inference(datos_val[i,], best_params)
      })
      
      current_val_error <- mean(abs((datos_val$MRR - pred_val) / datos_val$MRR))
      
      # parada anticipada
      if (current_val_error < best_overall_error) {
        best_overall_error <- current_val_error
        generations_no_improvement <- 0
      } else {
        generations_no_improvement <- generations_no_improvement + 1
      }
      
      if (generations_no_improvement >= patience) {
        break
      }
    }
    
    return(best_overall_error)
  }
  
  # Ejecutar cross-validation
  all_mape_results <- sapply(folds, run_fold)
  
  # Calcular MAPE promedio y desviación estándar
  avg_mape <- mean(all_mape_results)
  sd_mape <- sd(all_mape_results)
  
  cat("Cross-Validation Results:\n")
  cat(sprintf("Average MAPE: %.2f%%\n", avg_mape * 100))
  cat(sprintf("MAPE Standard Deviation: %.2f%%\n", sd_mape * 100))
  
  return(avg_mape)
}

# Preparación de datos
datos_norm <- prepare_data(datos)

# Ejecutar algoritmo genético 
resultado_final <- advanced_genetic_algorithm(datos_norm)

# Análisis de la calidad de predicción
cat("\nAnálisis de Calidad de Predicción:\n")
if (resultado_final <= 0.1) {
  cat("Excelente predicción (MAPE <= 10%)\n")
} else if (resultado_final <= 0.2) {
  cat("Predicción aceptable (10% < MAPE <= 20%)\n")
} else {
  cat("Predicción pobre (MAPE > 20%)\n")
}