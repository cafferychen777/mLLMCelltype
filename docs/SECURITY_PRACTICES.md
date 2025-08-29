# Security Practices for mLLMCelltype

## API Key Protection

### ❌ Never Do This:
```python
# DON'T: Hard-code API keys
api_key = "sk-proj-abc123..."
openai_key = "sk-proj-xyz789..."
```

### ✅ Always Do This:
```python
# DO: Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")
```

## Git Protection Measures

### 1. .gitignore Patterns
The repository includes comprehensive .gitignore patterns for:
- API key files (`*api_key*`, `*secret*`)
- Environment files (`*.env`)
- Test reports that might contain keys
- Archive directories

### 2. Pre-commit Hooks
A pre-commit hook automatically scans for:
- OpenAI API keys (`sk-proj-...`, `sk-...`)
- Anthropic API keys (`sk-ant-api03-...`)
- Google API keys (`AIzaSy...`)
- OpenRouter API keys (`sk-or-v1-...`)
- Bearer tokens
- Other sensitive patterns

### 3. Git Filters
Automatic filters clean sensitive data before commits.

## Best Practices

### For Development:
1. **Use environment variables**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. **Use .env files** (add to .gitignore):
   ```bash
   # .env
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   ```

3. **Use config files** (add to .gitignore):
   ```python
   # config/local.py
   API_KEYS = {
       "openai": "your-key-here",
       "anthropic": "your-key-here"
   }
   ```

### For Documentation:
- Use placeholder values: `"your-api-key-here"`
- Use redacted examples: `"sk-proj-[REDACTED]"`
- Reference environment variables instead of actual keys

### For Testing:
- Never commit test files with real API keys
- Use mock responses for unit tests
- Store test keys in environment variables only

## Security Tools

### Check for Sensitive Data:
```bash
# Run security scan
./scripts/check-sensitive.sh

# Check specific directory
./scripts/check-sensitive.sh python/examples/
```

### Before Committing:
1. Run the security check script
2. Review all files being committed
3. Ensure no real API keys are included

## Emergency Response

### If API Key is Accidentally Committed:
1. **Immediately revoke the key** in the provider's dashboard
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch path/to/file' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (⚠️ dangerous - coordinate with team):
   ```bash
   git push origin --force --all
   ```
4. **Generate new API key**
5. **Update all environments** with new key

## Monitoring

- GitHub has secret scanning enabled
- Pre-commit hooks provide first line of defense
- Regular security audits using the check script
- Team members should report any suspected key exposure immediately

## Contact

For security concerns, contact the repository maintainers immediately.
