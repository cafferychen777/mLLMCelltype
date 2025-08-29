#!/bin/bash

# Script to check for sensitive information in the repository
# Usage: ./scripts/check-sensitive.sh [directory]

SEARCH_DIR=${1:-.}

echo "🔍 Scanning for sensitive information in: $SEARCH_DIR"
echo "=================================================="

# API Key patterns
echo "Checking for API keys..."
grep -r -n --include="*.py" --include="*.R" --include="*.md" --include="*.json" --include="*.txt" \
    -E "(sk-[a-zA-Z0-9]{32,}|sk-ant-api03-[a-zA-Z0-9_-]{95,}|AIzaSy[a-zA-Z0-9_-]{33}|sk-or-v1-[a-zA-Z0-9]{64})" \
    "$SEARCH_DIR" 2>/dev/null | head -20

echo ""
echo "Checking for API key variables..."
grep -r -n --include="*.py" --include="*.R" --include="*.md" --include="*.json" \
    -i "api_key.*=.*['\"][^'\"]{20,}['\"]" \
    "$SEARCH_DIR" 2>/dev/null | head -10

echo ""
echo "Checking for environment variable assignments..."
grep -r -n --include="*.py" --include="*.R" --include="*.md" \
    -E "(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY).*=.*['\"][^'\"]{20,}['\"]" \
    "$SEARCH_DIR" 2>/dev/null | head -10

echo ""
echo "Checking for Bearer tokens..."
grep -r -n --include="*.py" --include="*.R" --include="*.md" --include="*.json" \
    -E "Bearer [a-zA-Z0-9_-]{32,}" \
    "$SEARCH_DIR" 2>/dev/null | head -10

echo ""
echo "✅ Scan complete. If no results shown above, no obvious sensitive data found."
echo ""
echo "💡 Tips:"
echo "- Always use environment variables for API keys"
echo "- Add sensitive files to .gitignore"
echo "- Use placeholder values in documentation"
echo "- Run this script before committing"
