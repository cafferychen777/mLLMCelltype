#!/usr/bin/env python3
"""Check that Vue template handlers and models exist in the admin application."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ADMIN_TEMPLATE = Path("templates/admin.html")
JS_KEYWORDS = {
    "catch",
    "confirm",
    "for",
    "if",
    "Number",
    "return",
    "switch",
    "while",
}


def extract_admin_script(content: str) -> str:
    """Return the inline script that contains the Vue application."""
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", content, re.DOTALL)
    for script in scripts:
        if "createApp({" in script:
            return script
    raise ValueError("Admin Vue script was not found")


def method_definitions(script: str) -> set[str]:
    """Extract shorthand method definitions from the Vue options object."""
    pattern = r"^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"
    return set(re.findall(pattern, script, re.MULTILINE)) - JS_KEYWORDS


def object_properties(script: str) -> set[str]:
    """Extract object keys used for Vue reactive data."""
    pattern = r"^\s*([A-Za-z_$][\w$]*)\s*:"
    return set(re.findall(pattern, script, re.MULTILINE))


def template_handlers(content: str) -> set[str]:
    """Extract method names referenced by Vue event directives."""
    expressions = re.findall(r'@[\w.:-]+\s*=\s*"([^"]+)"', content)
    handlers: set[str] = set()
    for expression in expressions:
        bare = re.fullmatch(r"\s*([A-Za-z_$][\w$]*)\s*", expression)
        if bare:
            handlers.add(bare.group(1))
        handlers.update(
            name
            for name in re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", expression)
            if name not in JS_KEYWORDS
        )
    return handlers


def template_models(content: str) -> set[str]:
    """Extract top-level properties referenced by v-model directives."""
    values = re.findall(r'v-model(?:\.\w+)*\s*=\s*"([^"]+)"', content)
    return {value.split(".", 1)[0].strip() for value in values}


def main() -> int:
    if not ADMIN_TEMPLATE.exists():
        print(f"Missing template: {ADMIN_TEMPLATE}", file=sys.stderr)
        return 1

    content = ADMIN_TEMPLATE.read_text(encoding="utf-8")
    try:
        script = extract_admin_script(content)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    methods = method_definitions(script)
    properties = object_properties(script)
    missing_handlers = template_handlers(content) - methods
    missing_models = template_models(content) - properties

    if missing_handlers:
        print("Undefined Vue handlers: " + ", ".join(sorted(missing_handlers)))
    if missing_models:
        print("Undefined v-model properties: " + ", ".join(sorted(missing_models)))
    if missing_handlers or missing_models:
        return 1

    print(
        f"Frontend contract check passed: {len(methods)} methods, "
        f"{len(template_models(content))} models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
