# Security Policy

## Supported Versions

mLLMCelltype is actively maintained with security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :x:                |
| < 1.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities in mLLMCelltype seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 Private Disclosure

Please **do not** report security vulnerabilities through public GitHub issues. Instead, report them privately to:

- **Email**: cafferychen777@gmail.com
- **Subject**: "[SECURITY] mLLMCelltype Vulnerability Report"

### 📋 What to Include

Please include the following information in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and affected versions
- Any suggested fixes (if available)

### ⏱️ Response Timeline

We will acknowledge your report within **48 hours** and provide:

- Initial assessment within 5 business days
- Regular updates on our progress
- Credit for responsible disclosure (if desired)

### 🛡️ Security Best Practices

When using mLLMCelltype:

- **API Keys**: Never commit API keys to your repository
- **Data Privacy**: Be mindful of sensitive biological data when using cloud LLM services
- **Network Security**: Use HTTPS endpoints for all API communications
- **Access Control**: Limit API key permissions to necessary scopes only

## Security Features

mLLMCelltype includes several security features:

- **API Key Management**: Secure storage and rotation support
- **Input Validation**: Sanitization of user inputs to prevent injection attacks
- **Error Handling**: Safe error messages that don't expose sensitive information
- **Audit Logging**: Comprehensive logging for security monitoring

## Contact

For general security questions about mLLMCelltype:
- GitHub Discussions: https://github.com/cafferychen777/mLLMCelltype/discussions
- Discord Community: https://discord.gg/pb2aZdG4