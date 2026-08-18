# .env 文件导入功能

## 功能概述

.env 文件导入功能允许用户通过上传一个 .env 文件来快速导入所有 API 密钥，无需逐个手动输入。

## 使用方法

1. **准备 .env 文件**
   - 创建一个名为 `.env` 的文本文件
   - 按照格式添加 API 密钥：`PROVIDER_API_KEY=你的密钥`
   - 可以下载示例文件作为参考

2. **上传文件**（两种方式）

   **方式一：点击上传**
   - 在配置步骤中，找到"导入 API 密钥"部分
   - 点击"上传 .env 文件"按钮
   - 选择你的 .env 文件

   **方式二：拖放上传**
   - 从文件管理器中拖动 .env 文件
   - 拖放到"导入 API 密钥"区域的任意位置
   - 鼠标悬停时区域会高亮显示

3. **自动导入**
   - 系统将自动：
     - 解析 .env 文件
     - 导入识别的 API 密钥
     - 自动选中有密钥的提供商
     - 显示成功/错误信息

## 支持的 API 密钥

系统识别以下环境变量名：

- `OPENAI_API_KEY` - OpenAI API 密钥
- `ANTHROPIC_API_KEY` - Anthropic Claude API 密钥
- `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` - Google Gemini API 密钥
- `GROK_API_KEY` 或 `XAI_API_KEY` - X.AI Grok API 密钥
- `DEEPSEEK_API_KEY` - DeepSeek API 密钥
- `QWEN_API_KEY` - 阿里通义千问 API 密钥
- `ZHIPU_API_KEY` 或 `GLM_API_KEY` - 智谱 GLM API 密钥
- `STEPFUN_API_KEY` - 阶跃星辰 API 密钥
- `MINIMAX_API_KEY` - MiniMax API 密钥
- `OPENROUTER_API_KEY` - OpenRouter API 密钥

## 示例 .env 文件

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# Add more as needed...
```

## 功能优势

1. **节省时间**：一次导入所有密钥，无需逐个复制粘贴
2. **减少错误**：避免手动输入时的拼写错误
3. **可重复使用**：同一个 .env 文件可在多次会话中使用
4. **便于管理**：将所有 API 密钥集中在一个安全文件中

## 安全说明

- API 密钥仅用于当前会话
- 密钥不会存储在服务器上
- .env 文件在浏览器端处理
- 请妥善保管 .env 文件，切勿分享

## 使用技巧

1. 下载示例 .env 文件作为模板
2. 将 .env 文件保存在安全位置
3. 获得新的 API 密钥时更新文件
4. 使用注释（以 # 开头的行）来组织文件内容
