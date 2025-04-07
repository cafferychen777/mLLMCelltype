# mLLMCelltype

mLLMCelltype是一个用于单细胞RNA测序数据细胞类型注释的多模态大语言模型框架。该项目同时支持R和Python接口，允许研究人员利用多种大语言模型（如GPT-4o、Claude、Gemini等）进行细胞类型注释。

## 主要特点

- **多LLM共识注释**：整合多个大语言模型的预测，提高注释可靠性
- **不确定性量化**：提供共识比例(CP)和熵(H)等指标评估注释可靠性
- **迭代讨论机制**：针对有争议的细胞群进行多轮讨论，提高注释准确性
- **高效缓存系统**：减少API调用，节省时间和成本
- **双语言支持**：同时提供R和Python接口，满足不同用户需求

## 目录结构

- `R/`：R语言接口和实现
- `python/`：Python接口和实现

## 安装

### R版本

```r
# 从GitHub安装
devtools::install_github("username/mLLMCelltype", subdir = "R")
```

### Python版本

```bash
# 从GitHub安装
pip install git+https://github.com/username/mLLMCelltype.git#subdirectory=python
```

## 使用示例

请参考各语言目录中的examples文件夹获取详细使用示例。

## 许可证

MIT
