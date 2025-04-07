library(testthat)
library(LLMCelltype)
library(withr) # 使用withr包来管理临时目录

# Mock data creation function
create_mock_data <- function() {
  # Create a list of genes for each cluster
  data <- list(
    "1" = list(
      genes = c("CD19", "CD20", "CD79A", "CD79B", "PAX5"),
      description = "B cell markers"
    ),
    "2" = list(
      genes = c("CD3D", "CD3E", "CD4", "CD8A", "IL7R"),
      description = "T cell markers"
    ),
    "3" = list(
      genes = c("CD14", "CD68", "CD163", "CSF1R", "ITGAM"),
      description = "Monocyte markers"
    )
  )
  return(data)
}

# Read and parse .env file
project_root <- "/Users/apple/Research/LLMCelltype"
env_lines <- readLines(file.path(project_root, ".env"))
for(line in env_lines) {
  if(grepl("^[[:alpha:]].*=.*$", line)) {
    parts <- strsplit(line, "=")[[1]]
    key <- trimws(parts[1])
    value <- trimws(gsub('"', '', parts[2]))
    do.call(Sys.setenv, structure(list(value), names = key))
  }
}

# Load API keys from environment variables
api_keys <- list(
  "claude-3-5-sonnet-latest" = Sys.getenv("ANTHROPIC_API_KEY"),
  "gemini-1.5-pro" = Sys.getenv("GEMINI_API_KEY"),
  "qwen-max-2025-01-25" = Sys.getenv("QWEN_API_KEY"),
  "gpt-4o" = Sys.getenv("OPENAI_API_KEY"),
  "deepseek-chat" = Sys.getenv("DEEPSEEK_API_KEY")
)

# Test cases
test_that("interactive_consensus_annotation handles input data correctly", {
  # 使用withr::with_tempdir创建临时目录并在测试结束后自动清理
  withr::with_tempdir({
    # 创建临时日志目录
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE, showWarnings = FALSE)
    
    # 创建mock数据
    mock_data <- create_mock_data()
    
    # 测试模型
    test_models <- c(
      "claude-3-5-sonnet-latest",  # Anthropic
      "gemini-1.5-pro",           # Google
      "qwen-max-2025-01-25"       # Qwen
    )
    
    # 运行函数
    result <- tryCatch({
      interactive_consensus_annotation(
        input = mock_data,
        tissue_name = "PBMC",
        models = test_models,
        api_keys = api_keys,
        top_gene_count = 5,
        controversy_threshold = 0.7,
        log_dir = temp_log_dir
      )
    }, error = function(e) {
      message("Error occurred during annotation: ", e$message)
      e
    })
    
    # 基本结构测试
    expect_false(inherits(result, "error"))
    expect_type(result, "list")
    
    # 检查日志文件是否创建
    expect_true(dir.exists(temp_log_dir))
    expect_true(length(list.files(temp_log_dir)) > 0)
    
    # 不需要手动清理，withr::with_tempdir会自动清理
  })
})



test_that("DiscussionLogger works correctly", {
  # 使用withr::with_tempdir创建临时目录并在测试结束后自动清理
  withr::with_tempdir({
    # 创建临时日志目录
    temp_log_dir <- file.path(getwd(), "test_logs")
    dir.create(temp_log_dir, recursive = TRUE, showWarnings = FALSE)
    
    # 初始化logger
    logger <- DiscussionLogger$new(temp_log_dir)
    
    # 测试日志记录
    cluster_id <- 1
    tissue_name <- "PBMC"
    marker_genes <- c("CD19", "CD20", "CD79A")
    
    # 开始集群讨论
    logger$start_cluster_discussion(cluster_id, tissue_name, marker_genes)
    
    # 记录事件
    logger$log_prediction("claude", 1, list(cell_type = "B cell", confidence = 0.9))
    logger$log_consensus_check(1, TRUE, 0.9)
    logger$log_final_consensus("B cell", "High confidence B cell annotation")
    
    # 结束讨论
    logger$end_cluster_discussion()
    
    # 检查日志文件是否存在
    log_files <- list.files(temp_log_dir, recursive = TRUE)
    expect_true(length(log_files) > 0)
    
    # 不需要手动清理，withr::with_tempdir会自动清理
  })
})
