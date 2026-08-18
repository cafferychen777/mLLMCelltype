# Template Syntax Guide

## ⚠️ CRITICAL: Custom Jinja2 Delimiters

This application uses **CUSTOM Jinja2 delimiters** to avoid conflicts with Vue.js:

### ✅ CORRECT Syntax

```html
<!-- Use {[{ }]} for all Jinja2 template expressions -->
<link rel="stylesheet" href="{[{ url_for('static', filename='css/style.css') }]}">
<a href="{[{ url_for('index') }]}">Home</a>
<script src="{[{ url_for('static', filename='js/app.js') }]}"></script>
```

### ❌ WRONG Syntax

```html
<!-- DO NOT use {{ }} - this conflicts with Vue.js -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
<a href="{{ url_for('index') }}">Home</a>
<script src="{{ url_for('static', filename='js/app.js') }}"></script>
```

## Why Custom Delimiters?

1. **Vue.js Conflict**: Vue.js uses `{{ }}` for its template interpolation
2. **Parsing Issues**: Using standard `{{ }}` causes JavaScript syntax errors
3. **White Screen Bug**: Incorrect delimiters prevent CSS/JS files from loading

## Configuration Location

The custom delimiters are configured in `app.py`:

```python
# Configure Jinja2 to avoid conflicts with Vue.js
# IMPORTANT: Use {[{ }]} instead of {{ }} in ALL template files
# This prevents conflicts with Vue.js template syntax
# DO NOT change this to {{ }} - it will break Vue.js functionality
app.jinja_env.variable_start_string = '{[{'
app.jinja_env.variable_end_string = '}]}'
```

## Common Mistakes to Avoid

### 1. Mixing Syntaxes

```html
<!-- WRONG: Mixing both syntaxes -->
<link rel="stylesheet" href="{[{ url_for('static', filename='css/style.css') }]}">
<a href="{{ url_for('index') }}">Home</a>  <!-- This will break! -->
```

### 2. Copy-Pasting from Other Flask Projects

When copying code from other Flask projects, remember to convert `{{ }}` to `{[{ }]}`.

### 3. IDE Auto-completion

Some IDEs might suggest `{{ }}` syntax. Always use `{[{ }]}` instead.

## Template Files Using This Syntax

All template files in the `templates/` directory use this custom syntax:

- `templates/index.html`
- `templates/admin.html`
- `templates/admin-login.html`
- `templates/how_it_works.html`
- `templates/how_to_annotate.html`
- `templates/annotation_accuracy.html`
- `templates/404.html`
- `templates/test.html`

## Debugging Template Issues

If you encounter white screen or JavaScript errors:

1. **Check Console**: Look for "Invalid or unexpected token" errors
2. **Verify Syntax**: Ensure all `url_for()` calls use `{[{ }]}`
3. **Check Network Tab**: See if CSS/JS files are loading (404 errors indicate template issues)

## Quick Fix Command

To find and fix incorrect syntax:

```bash
# Find files with incorrect syntax
grep -r "{{ url_for" templates/

# Should return no results if all files use correct syntax
```

## Fixed Issues

### Admin White Screen Bug (2025-07-07)

**Problem**: Another LLM incorrectly changed Vue.js delimiters from `{{ }}` to `[[ ]]`, causing JavaScript syntax errors and white screen.

**Root Cause**: The other LLM misunderstood the solution and modified Vue.js instead of keeping Jinja2 custom delimiters.

**Correct Solution**:

- ✅ Keep Vue.js using standard `{{ }}` delimiters
- ✅ Keep Jinja2 using custom `{[{ }]}` delimiters
- ✅ Fixed with `scripts/fix_vue_delimiters.py`

## Remember

- **Always use `{[{ }]}` for Jinja2 expressions**
- **Always use `{{ }}` for Vue.js expressions**
- **Never change Vue.js delimiters - this breaks the framework**
- **This prevents Vue.js conflicts and white screen issues**
- **When in doubt, check existing working templates for reference**
