# 使用用户数据进行小鼠肾脏细胞类型注释测试
# 基于 mLLMCelltype 包的多模型共识注释

# 加载必要的包
library(mLLMCelltype)

# 创建缓存和日志目录 - 指明具体目录路径
cache_dir <- "/Users/apple/Research/mLLMCelltype/R/examples/cache/mice_kidney_updated"
log_dir <- "/Users/apple/Research/mLLMCelltype/R/examples/logs/mice_kidney_updated"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(log_dir, showWarnings = FALSE, recursive = TRUE)

cat("缓存目录:", cache_dir, "\n")
cat("日志目录:", log_dir, "\n")

# 加载用户的数据文件
markers_file <- "/Users/apple/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/97b7f0e43b31cf0698c7e1c1db358c0b/Message/MessageTemp/b86b3c4eb01e5ea1dcea9129990094bf/File/markers_2.rda"

cat("正在加载数据文件:", markers_file, "\n")
load(markers_file)

# 检查加载的数据
cat("加载的对象:\n")
print(ls())

# 检查 markers_2 对象并重命名为 markers
if (exists("markers_2")) {
  markers <- markers_2
  cat("\nmarkers_2 对象信息 (重命名为 markers):\n")
  cat("类型:", class(markers), "\n")
  if (is.list(markers)) {
    cat("集群数量:", length(markers), "\n")
    cat("前5个集群名称:", paste(names(markers)[1:min(5, length(markers))], collapse=", "), "\n")

    # 显示前几个集群的基因信息
    for (i in 1:min(3, length(markers))) {
      cluster_name <- names(markers)[i]
      if (is.list(markers[[i]]) && "genes" %in% names(markers[[i]])) {
        genes <- markers[[i]]$genes
      } else if (is.character(markers[[i]])) {
        genes <- markers[[i]]
      } else {
        genes <- "未知格式"
      }
      cat(sprintf("集群 %s: %s\n", cluster_name,
                  ifelse(is.character(genes),
                         paste(head(genes, 5), collapse=", "),
                         "非字符向量")))
    }
  } else {
    cat("markers_2 不是列表格式\n")
    print(str(markers))
  }
} else {
  cat("未找到 markers_2 对象，可用对象:\n")
  print(ls())
}

# 设置 API 密钥
api_keys <- list(
  openrouter = "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"
)

# 如果 markers 对象存在且格式正确，运行注释
if (exists("markers") && is.list(markers) && length(markers) > 0) {
  cat("\n开始使用 OpenRouter 免费模型进行细胞类型注释...\n")

  # 运行共识注释
  consensus_results <- interactive_consensus_annotation(
    input = markers,
    tissue_name = "mice kidney",
    models = c(
      "meta-llama/llama-4-maverick:free",           # Meta Llama 4 Maverick (free)
      "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", # NVIDIA Nemotron Ultra 253B (free)
      "microsoft/mai-ds-r1:free",                   # Microsoft MAI-DS-R1 (free)
      "deepseek/deepseek-chat-v3-0324:free"        # DeepSeek Chat v3 (free)
    ),
    api_keys = api_keys,
    consensus_check_model = "deepseek/deepseek-chat-v3-0324:free",
    controversy_threshold = 0.7,
    entropy_threshold = 1.0,
    max_discussion_rounds = 2,
    cache_dir = cache_dir,
    log_dir = log_dir
  )

  # 保存结果 - 指明具体文件名
  results_file <- "/Users/apple/Research/mLLMCelltype/R/examples/mice_kidney_updated_results.rds"
  saveRDS(consensus_results, results_file)
  cat("\n结果已保存至:", results_file, "\n")

  # 保存为 CSV 格式，方便查看 - 指明具体文件名
  results_csv <- "/Users/apple/Research/mLLMCelltype/R/examples/mice_kidney_updated_results.csv"
  final_annotations_df <- data.frame(
    Cluster = names(consensus_results$final_annotations),
    CellType = unlist(consensus_results$final_annotations),
    stringsAsFactors = FALSE
  )
  write.csv(final_annotations_df, results_csv, row.names = FALSE)
  cat("结果已保存为 CSV 格式:", results_csv, "\n")

  # 打印结果摘要
  cat("\n结果摘要:\n")
  cat("可用字段:", paste(names(consensus_results), collapse=", "), "\n\n")

  # 打印最终注释
  cat("最终细胞类型注释:\n")
  for (cluster in names(consensus_results$final_annotations)) {
    cat(sprintf("%s: %s\n", cluster, consensus_results$final_annotations[[cluster]]))
  }

  # 打印有争议的集群
  cat("\n有争议的集群:", paste(consensus_results$controversial_clusters, collapse=", "), "\n")

  # 检查集群数量
  cat("\n集群数量检查:\n")
  cat("输入集群数量:", length(markers), "\n")
  cat("最终注释集群数量:", length(consensus_results$final_annotations), "\n")
  cat("有争议集群数量:", length(consensus_results$controversial_clusters), "\n")

} else {
  cat("错误: markers 对象不存在或格式不正确，无法运行注释\n")
}
