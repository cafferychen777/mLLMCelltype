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
  # Print consensus building summary
  cat("\nConsensus Building Summary:\n")
  cat(sprintf("Total clusters analyzed: %d\n", length(results$final_annotations)))
  cat(sprintf("Controversial clusters requiring discussion: %d\n", 
              length(results$controversial_clusters)))
  
  # If there are controversial clusters, print detailed results
  if (length(results$controversial_clusters) > 0) {
    cat("\nDetailed results for controversial clusters:\n")
    
    # Iterate through each controversial cluster
    for (cluster_id in results$controversial_clusters) {
      char_cluster_id <- as.character(cluster_id)
      numeric_cluster_id <- as.numeric(cluster_id)
      
      cat(sprintf("\nCluster %s:\n", char_cluster_id))
      cat("Initial predictions:\n")
      
      # 优先使用discussion_logs中的初始预测，因为这些是实际用于讨论的预测
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
        
        # Check if predictions have names
        first_model <- names(results$initial_results$individual_predictions)[1]
        predictions <- results$initial_results$individual_predictions[[first_model]]
        has_names <- !is.null(names(predictions))
        
        # Iterate through each model, get and print its prediction
        for (model in names(results$initial_results$individual_predictions)) {
          prediction <- tryCatch({
            # Access prediction differently based on whether it has names
            if (has_names) {
              # 使用[[]]访问列表中的元素
              pred <- results$initial_results$individual_predictions[[model]][[char_cluster_id]]
            } else {
              # 如果没有名称，尝试不同的方式获取预测
              model_predictions <- results$initial_results$individual_predictions[[model]]
              
              # 检查model_predictions的类型
              if (is.list(model_predictions)) {
                # 如果是列表，尝试使用数值索引
                if (numeric_cluster_id <= length(model_predictions)) {
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
            
            # 清理预测结果，去除可能的前缀（如"19: "）
            if (!is.null(pred) && !is.na(pred) && pred != "") {
              pred <- gsub("^[0-9]+:[[:space:]]*", "", pred)  # 移除聚类ID前缀
              pred <- gsub("[[:space:]]+$", "", pred)  # 移除尾部空格
            }
            
            # Handle empty or NA predictions
            if (is.null(pred) || is.na(pred) || pred == "") {
              "未提供预测"
            } else {
              pred
            }
          }, error = function(e) {
            # Handle errors when retrieving prediction
            "检索预测时出错"
          })
          
          cat(sprintf("  %s: %s\n", model, prediction))
        }
      }
      } else {
        cat("  Initial predictions not available\n")
      }
      
      # Print uncertainty metrics
      if (!is.null(results$initial_results) && 
          !is.null(results$initial_results$consensus_results) &&
          !is.null(results$initial_results$consensus_results[[char_cluster_id]])) {
        
        consensus_result <- results$initial_results$consensus_results[[char_cluster_id]]
        
        cat("\nUncertainty metrics:\n")
        
        # Print consensus proportion
        if (!is.null(consensus_result$consensus_proportion)) {
          cat(sprintf("  Consensus proportion: %.2f\n", consensus_result$consensus_proportion))
        } else if (!is.null(consensus_result$agreement_score)) {
          # If no consensus_proportion but agreement_score exists, use it as alternative
          cat(sprintf("  Agreement score: %.2f\n", consensus_result$agreement_score))
        }
        
        # Print entropy
        if (!is.null(consensus_result$entropy)) {
          cat(sprintf("  Shannon entropy: %.2f\n", consensus_result$entropy))
        }
      } else {
        cat("\nUncertainty metrics: Not available\n")
      }
      
      # Print final consensus
      if (!is.null(results$final_annotations) && 
          !is.null(results$final_annotations[[char_cluster_id]])) {
        
        final_annotation <- results$final_annotations[[char_cluster_id]]
        
        # Process final_annotation differently based on its type
        if (is.list(final_annotation) || (is.vector(final_annotation) && length(final_annotation) > 0 && !is.character(final_annotation))) {
          # If it's a list or non-character vector, take the first element
          final_annotation_str <- as.character(final_annotation[[1]])
        } else {
          # Otherwise convert directly to string
          final_annotation_str <- as.character(final_annotation)
        }
        
        # 验证最终共识与初始预测的一致性
        if (!is.null(results$initial_results) && 
            !is.null(results$initial_results$individual_predictions)) {
          
          # 收集所有模型的预测
          all_predictions <- list()
          for (model in names(results$initial_results$individual_predictions)) {
            if (has_names) {
              # 使用[[]]访问列表中的元素
              pred <- results$initial_results$individual_predictions[[model]][[char_cluster_id]]
            } else {
              # 如果没有名称，尝试不同的方式获取预测
              model_predictions <- results$initial_results$individual_predictions[[model]]
              
              # 检查model_predictions的类型
              if (is.list(model_predictions)) {
                # 如果是列表，尝试使用数值索引
                if (numeric_cluster_id <= length(model_predictions)) {
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
            
            if (!is.null(pred) && !is.na(pred) && pred != "") {
              all_predictions[[model]] <- pred
            }
          }
          
          # 检查是否所有模型都预测相同的结果
          if (length(all_predictions) > 0) {
            unique_preds <- unique(unlist(all_predictions))
            if (length(unique_preds) == 1) {
              # 清理预测结果，去除可能的前缀（如"19: "）
              clean_pred <- gsub("^[0-9]+:\\s*", "", unique_preds[1])
              clean_pred <- gsub("\\s+$", "", clean_pred)  # 移除尾部空格
              
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
  } else {
    cat("\nNo controversial clusters found. All clusters reached consensus.\n")
  }
}
