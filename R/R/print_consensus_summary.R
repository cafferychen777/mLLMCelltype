#' Print summary of consensus results
#' 
#' This function prints a detailed summary of the consensus building process,
#' including initial predictions from all models, uncertainty metrics, and final consensus
#' for each controversial cluster.
#' 
#' @param results A list containing consensus annotation results with the following components:
#'   \itemize{
#'     \item initial_results: A list containing individual_predictions, consensus_results, and controversial_clusters
#'     \item final_annotations: A list of final cell type annotations for each cluster
#'     \item controversial_clusters: A character vector of cluster IDs that were controversial
#'     \item discussion_logs: A list of discussion logs for each controversial cluster
#'   }
#' @return None, prints summary to console
#' @keywords internal
print_consensus_summary <- function(results) {
  # Debug: Print the structure of results
  cat("DEBUG: Structure of results:\n")
  cat(sprintf("DEBUG: Names in results: %s\n", paste(names(results), collapse = ", ")))
  cat(sprintf("DEBUG: Length of final_annotations: %d\n", length(results$final_annotations)))
  cat(sprintf("DEBUG: Length of controversial_clusters: %d\n", length(results$controversial_clusters)))
  
  # Print consensus building summary
  cat("\nConsensus Building Summary:\n")
  cat(sprintf("Total clusters analyzed: %d\n", length(results$final_annotations)))
  cat(sprintf("Controversial clusters requiring discussion: %d\n", 
              length(results$controversial_clusters)))
  
  # If there are controversial clusters, print detailed results
  if (length(results$controversial_clusters) > 0) {
    # Debug: Print controversial clusters
    cat("DEBUG: Controversial clusters: ", paste(results$controversial_clusters, collapse = ", "), "\n")
    cat("\nDetailed results for controversial clusters:\n")
    
    # Iterate through each controversial cluster
    for (cluster_id in results$controversial_clusters) {
      # Debug: Print current cluster ID
      cat(sprintf("DEBUG: Processing cluster_id: %s (type: %s)\n", cluster_id, typeof(cluster_id)))
      char_cluster_id <- as.character(cluster_id)
      numeric_cluster_id <- as.numeric(cluster_id)
      
      cat(sprintf("\nCluster %s:\n", char_cluster_id))
      cat("Initial predictions:\n")
      
      # 优先使用discussion_logs中的初始预测，因为这些是实际用于讨论的预测
      cat("DEBUG: Checking discussion_logs...\n")
      cat(sprintf("DEBUG: discussion_logs exists: %s\n", !is.null(results$discussion_logs)))
      if (!is.null(results$discussion_logs)) {
        cat(sprintf("DEBUG: discussion_logs[[%s]] exists: %s\n", char_cluster_id, !is.null(results$discussion_logs[[char_cluster_id]])))
      }
      if (!is.null(results$discussion_logs) && !is.null(results$discussion_logs[[char_cluster_id]])) {
        cat(sprintf("DEBUG: discussion_logs[[%s]]$initial_predictions exists: %s\n", char_cluster_id, !is.null(results$discussion_logs[[char_cluster_id]]$initial_predictions)))
      }
      
      if (!is.null(results$discussion_logs) && 
          !is.null(results$discussion_logs[[char_cluster_id]]) && 
          !is.null(results$discussion_logs[[char_cluster_id]]$initial_predictions)) {
        
        # 使用讨论日志中的初始预测
        initial_predictions <- results$discussion_logs[[char_cluster_id]]$initial_predictions
        
        # 遍历每个模型的预测
        for (model in names(initial_predictions)) {
          prediction <- initial_predictions[[model]]
          
          # 处理空或NA预测
          if (is.null(prediction) || is.na(prediction) || prediction == "") {
            prediction <- "未提供预测"
          }
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      } 
      # 如果讨论日志中没有初始预测，则使用initial_results
      else if (!is.null(results$initial_results) && 
               !is.null(results$initial_results$individual_predictions)) {
        cat("DEBUG: Using initial_results$individual_predictions\n")
        cat(sprintf("DEBUG: Models in individual_predictions: %s\n", paste(names(results$initial_results$individual_predictions), collapse = ", ")))
        
        # 遍历每个模型的预测
        for (model in names(results$initial_results$individual_predictions)) {
          # 检查预测是否有名称
          first_model <- names(results$initial_results$individual_predictions)[1]
          predictions <- results$initial_results$individual_predictions[[first_model]]
          has_names <- !is.null(names(predictions))
          
          if (has_names) {
            # 如果有名称，使用字符串索引
            if (char_cluster_id %in% names(results$initial_results$individual_predictions[[model]])) {
              prediction <- results$initial_results$individual_predictions[[model]][[char_cluster_id]]
            } else {
              prediction <- NA
            }
          } else {
            # 如果没有名称，尝试使用数值索引
            if (numeric_cluster_id <= length(results$initial_results$individual_predictions[[model]])) {
              prediction <- results$initial_results$individual_predictions[[model]][numeric_cluster_id]
            } else {
              prediction <- NA
            }
          }
          
          # 处理空或NA预测
          if (is.null(prediction) || is.na(prediction) || prediction == "") {
            prediction <- "未提供预测"
          }
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      } else {
        cat("  No initial predictions available\n")
      }
      
      # 打印不确定性指标
      cat("\nUncertainty metrics:\n")
      if (!is.null(results$initial_results) && 
          !is.null(results$initial_results$consensus_results) && 
          !is.null(results$initial_results$consensus_results[[char_cluster_id]])) {
        
        consensus_result <- results$initial_results$consensus_results[[char_cluster_id]]
        
        # 打印共识比例
        if (!is.null(consensus_result$consensus_proportion)) {
          cat(sprintf("  Consensus proportion: %.2f\n", consensus_result$consensus_proportion))
        }
        
        # 打印香农熵
        if (!is.null(consensus_result$entropy)) {
          cat(sprintf("  Shannon entropy: %.2f\n", consensus_result$entropy))
        }
      } else {
        cat("  Uncertainty metrics not available\n")
      }
      
      # 打印最终共识
      cat("DEBUG: Checking final_annotations...\n")
      cat(sprintf("DEBUG: final_annotations exists: %s\n", !is.null(results$final_annotations)))
      if (!is.null(results$final_annotations)) {
        cat(sprintf("DEBUG: final_annotations[[%s]] exists: %s\n", char_cluster_id, !is.null(results$final_annotations[[char_cluster_id]])))
      }
      
      if (!is.null(results$final_annotations) && 
          !is.null(results$final_annotations[[char_cluster_id]])) {
        
        final_annotation <- results$final_annotations[[char_cluster_id]]
        cat(sprintf("DEBUG: final_annotation type: %s, is.list: %s, is.vector: %s, length: %d\n", 
                    typeof(final_annotation), 
                    is.list(final_annotation), 
                    is.vector(final_annotation), 
                    length(final_annotation)))
        
        cat(sprintf("DEBUG: Processing final_annotation - is.list: %s, is.vector: %s, length > 0: %s, !is.character: %s\n", 
                    is.list(final_annotation), 
                    is.vector(final_annotation), 
                    length(final_annotation) > 0, 
                    ifelse(is.vector(final_annotation), !is.character(final_annotation), TRUE)))
        
        if (is.list(final_annotation) || (is.vector(final_annotation) && length(final_annotation) > 0 && !is.character(final_annotation))) {
          # If it's a list or non-character vector, take the first element
          cat("DEBUG: Converting list/vector element to string\n")
          final_annotation_str <- tryCatch({
            as.character(final_annotation[[1]])
          }, error = function(e) {
            cat(sprintf("DEBUG: Error converting final_annotation[[1]] to string: %s\n", e$message))
            "Error converting to string"
          })
          cat(sprintf("DEBUG: final_annotation_str: %s\n", final_annotation_str))
        } else {
          # Otherwise convert directly to string
          final_annotation_str <- tryCatch({
            as.character(final_annotation)
          }, error = function(e) {
            cat(sprintf("DEBUG: Error converting final_annotation to string: %s\n", e$message))
            "Error converting to string"
          })
          cat(sprintf("DEBUG: final_annotation_str: %s\n", final_annotation_str))
        }
        
        # 验证最终共识与初始预测的一致性
        if (!is.null(results$initial_results) && 
            !is.null(results$initial_results$individual_predictions) &&
            length(names(results$initial_results$individual_predictions)) > 0) { # 确保有模型
          
          # 尝试获取第一个模型名称
          tryCatch({
            first_model <- names(results$initial_results$individual_predictions)[1]
            predictions <- results$initial_results$individual_predictions[[first_model]]
            has_names <- !is.null(names(predictions))
          }, error = function(e) {
            cat(sprintf("DEBUG: Error getting first model predictions: %s\n", e$message))
            has_names <- FALSE
          })
          
          # 收集所有模型的预测
          all_predictions <- list()
          cat("DEBUG: Collecting all model predictions for cluster_id: ", char_cluster_id, "\n")
          
          # 首先检查是否有模型预测
          if (length(names(results$initial_results$individual_predictions)) == 0) {
            cat("DEBUG: No models found in initial_results$individual_predictions\n")
            # 如果没有模型预测，尝试使用discussion_logs中的预测
            if (!is.null(results$discussion_logs) && 
                !is.null(results$discussion_logs[[char_cluster_id]]) && 
                !is.null(results$discussion_logs[[char_cluster_id]]$initial_predictions)) {
              
              cat("DEBUG: Using discussion_logs for predictions\n")
              initial_predictions <- results$discussion_logs[[char_cluster_id]]$initial_predictions
              
              for (model in names(initial_predictions)) {
                prediction <- initial_predictions[[model]]
                if (!is.null(prediction) && !is.na(prediction) && prediction != "" && prediction != "未提供预测") {
                  cat(sprintf("DEBUG: Adding prediction from discussion_logs for model %s: %s\n", model, prediction))
                  all_predictions[[model]] <- prediction
                }
              }
            }
          } else {
            # 正常处理initial_results中的预测
            for (model in names(results$initial_results$individual_predictions)) {
              if (has_names) {
                # 使用[[]]访问列表中的元素
                cat(sprintf("DEBUG: Accessing prediction with names for cluster %s\n", char_cluster_id))
                pred <- results$initial_results$individual_predictions[[model]][[char_cluster_id]]
              } else {
                # 如果没有名称，尝试不同的方式获取预测
                cat(sprintf("DEBUG: Accessing prediction without names for cluster %s (numeric: %d)\n", char_cluster_id, numeric_cluster_id))
                model_predictions <- results$initial_results$individual_predictions[[model]]
                
                # 检查model_predictions的类型
                if (is.list(model_predictions)) {
                  # 首先尝试使用字符串索引，这样可以避免索引混淆
                  if (char_cluster_id %in% names(model_predictions)) {
                    cat(sprintf("DEBUG: Accessing list with char_cluster_id: %s\n", char_cluster_id))
                    pred <- model_predictions[[char_cluster_id]]
                  } 
                  # 如果字符串索引不存在，再尝试使用数值索引
                  else if (numeric_cluster_id <= length(model_predictions)) {
                    cat(sprintf("DEBUG: Accessing list with numeric_cluster_id: %d\n", numeric_cluster_id))
                    pred <- model_predictions[[numeric_cluster_id]]
                  } else {
                    pred <- NA
                  }
                } else if (is.vector(model_predictions)) {
                  # 如果是向量，尝试使用数值索引
                  if (numeric_cluster_id <= length(model_predictions)) {
                    pred <- model_predictions[numeric_cluster_id]
                  } else {
                    pred <- NA
                  }
                } else {
                  # 其他情况
                  pred <- NA
                }
              }
              
              # 先检查是否为NA或NULL，然后再检查是否为空字符串
              if (!is.null(pred)) {
                if (!is.na(pred)) {
                  if (pred != "") {
                    # 确保我们添加的是当前聚类的预测，而不是其他聚类的
                    # 检查预测中是否包含聚类前缀
                    pred_cluster_prefix <- gsub("^([0-9]+):.*$", "\\1", pred)
                    
                    # 如果预测包含聚类前缀，确保它与当前聚类匹配
                    if (pred_cluster_prefix != pred) { # 如果有前缀
                      if (pred_cluster_prefix == char_cluster_id) {
                        cat(sprintf("DEBUG: Adding prediction for model %s: %s\n", model, pred))
                        all_predictions[[model]] <- pred
                      } else {
                        cat(sprintf("DEBUG: Skipping prediction for model %s: %s (wrong cluster prefix, expected %s)\n", 
                                    model, pred, char_cluster_id))
                      }
                    } else { # 如果没有前缀，直接添加
                      cat(sprintf("DEBUG: Adding prediction for model %s: %s\n", model, pred))
                      all_predictions[[model]] <- pred
                    }
                  }
                }
              }
            }
          }
          
          # 检查是否所有模型都预测相同的结果
          cat(sprintf("DEBUG: all_predictions length: %d\n", length(all_predictions)))
          if (length(all_predictions) > 0) {
            unique_preds <- unique(unlist(all_predictions))
            cat(sprintf("DEBUG: unique_preds: %s\n", paste(unique_preds, collapse = ", ")))
            if (length(unique_preds) == 1) {
              # 清理预测结果，去除可能的前缀（如"19: "）
              clean_pred <- gsub("^[0-9]+:[[:space:]]*", "", unique_preds[1])
              clean_pred <- gsub("[[:space:]]+$", "", clean_pred)  # 移除尾部空格
              
              # 如果所有模型预测相同但与最终共识不同，添加警告
              if (!grepl(clean_pred, final_annotation_str, ignore.case = TRUE) && 
                  !grepl(final_annotation_str, clean_pred, ignore.case = TRUE)) {
                cat(sprintf("WARNING: All models predicted '%s' but final consensus is '%s'\n", 
                            clean_pred, final_annotation_str))
              }
            }
          }
        }
        
        cat(sprintf("Final consensus: %s\n", final_annotation_str))
      } else {
        cat("Final consensus: Not available\n")
      }
    }
  }
  
  # 如果没有争议性聚类，打印信息
  if (length(results$controversial_clusters) == 0) {
    cat("\nNo controversial clusters found. All clusters reached consensus.\n")
  }
}