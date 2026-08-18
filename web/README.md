# mLLMCelltype Web

A web interface for the [mLLMCelltype](https://github.com/cafferychen777/mLLMCelltype) package, enabling easy cell type annotation using multiple large language models.

[![Paper](https://img.shields.io/badge/Commun.%20Biol.-10.1038%2Fs42003--026--10420--8-blue)](https://doi.org/10.1038/s42003-026-10420-8)
[![GitHub](https://img.shields.io/github/stars/cafferychen777/mLLMCelltype?style=social)](https://github.com/cafferychen777/mLLMCelltype)
[![Website](https://img.shields.io/badge/Website-mllmcelltype.com-green)](https://www.mllmcelltype.com)

## Overview

mLLMCelltype Web provides a user-friendly interface for annotating cell types in single-cell RNA sequencing data using multiple large language models. It supports various LLM providers including OpenAI, Anthropic, Google (Gemini), and many Chinese providers.

## Features

- **Simple File Upload**: Support for CSV, TSV, and Excel formats
- **Multiple LLM Providers**: 10 providers including OpenAI, Anthropic, Google, DeepSeek, and Qwen
- **Consensus Annotation**: Combine results from multiple models for higher accuracy
- **Interactive Discussion**: Models can discuss and refine annotations
- **Easy Result Download**: Export results in various formats
- **API Key Validation**: Test API keys before starting annotation
- **Admin Dashboard**: Monitor usage and task history
- **Multi-language Support**: Available in English and Chinese

## Architecture

- **Hetzner VPS**: Immutable Docker image releases
- **Host Caddy**: Independent reverse proxy with automatic TLS
- **Turso Database**: Distributed SQLite for task storage
- **Cloudflare DNS**: Domain management isolated from the application host
- **GitHub Actions**: CI/CD pipeline (`../.github/workflows/web-deploy-vps.yml`)
- **Custom Domain**: <https://www.mllmcelltype.com>

## Live Demo

Visit [https://www.mllmcelltype.com](https://www.mllmcelltype.com) to try the tool without installation.

## Quick Start

### For Users

1. Visit [https://www.mllmcelltype.com](https://www.mllmcelltype.com)
2. Upload your marker gene CSV file
3. Select models and enter API keys
4. Configure parameters
5. Start annotation and download results

### For Developers

> **⚠️ Important**: This app uses custom Jinja2 delimiters `{[{ }]}` instead of `{{ }}` to avoid Vue.js conflicts. See [docs/TEMPLATE_SYNTAX_GUIDE.md](docs/TEMPLATE_SYNTAX_GUIDE.md) for details.

#### Local Development

```bash
# Clone the repository
git clone https://github.com/cafferychen777/mLLMCelltype.git
cd mLLMCelltype/web

# Install dependencies
pip install --require-hashes -r requirements-dev.lock

# Copy and configure environment
cp .env.example .env
# Edit .env with Turso, Flask, and optional admin credentials

# Validate and run locally
make check
python app.py
```

#### Production Deployment

Pushing to `main` triggers a GitHub Actions deploy to the VPS:

```bash
git push origin main
```

CI builds the exact commit into a tagged image, transfers a checksummed release
bundle, and invokes the VPS's single scoped deployment command. The server never
pulls source code or builds production images. See [Deployment](docs/DEPLOYMENT.md)
for provisioning and rollback procedures.

## Configuration

### Environment Variables

See `.env.example` for the full list. Key variables:

- `TURSO_DB_URL`: Turso database URL
- `TURSO_AUTH_TOKEN`: Turso authentication token
- `FLASK_SECRET_KEY`: Stable random key used to sign ownership and admin sessions
- `BACKGROUND_THREADS_ENABLED`: Enables task monitoring and memory cleanup
- `ADMIN_USERNAME`: Admin dashboard username
- `ADMIN_PASSWORD_HASH`: Hashed admin password

### Supported Providers

OpenAI, Anthropic, Gemini, DeepSeek, Grok, Qwen, Zhipu, StepFun,
MiniMax, and OpenRouter are supported. The current selectable models and defaults
are maintained in `config/model_catalog.py`; the API and web UI both consume this
single catalog. Users can also add a provider-specific model ID in the interface.

## Documentation

- [Template Syntax Guide](docs/TEMPLATE_SYNTAX_GUIDE.md)
- [Development Guide](docs/DEVELOPMENT_GUIDE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Env File Import](docs/features/ENV_FILE_IMPORT.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the repository-wide MIT License - see
[LICENSE](../LICENSE) for details.

## Citation

If you use mLLMCelltype in your research, please cite:

```bibtex
@article{yang2026llmconsensus,
  author = {Yang, Chen and Zhang, Xianyang and Chen, Jun},
  title = {Large language model consensus substantially improves the cell type annotation accuracy for scRNA-seq data},
  journal = {Communications Biology},
  year = {2026},
  volume = {9},
  pages = {779},
  doi = {10.1038/s42003-026-10420-8},
  publisher = {Nature Publishing Group}
}
```

## Contact

- Website: <https://www.mllmcelltype.com>
- GitHub: <https://github.com/cafferychen777/mllmcelltype>
- Email: <cafferychen@gmail.com>
