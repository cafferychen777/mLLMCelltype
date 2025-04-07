#!/usr/bin/env python3
import os
import re
import sys

def update_imports(directory):
    """
    Update all import statements from 'llmcelltype' to 'mllmcelltype' in Python files
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace import statements
                updated_content = re.sub(r'import llmcelltype', 'import mllmcelltype', content)
                updated_content = re.sub(r'from llmcelltype', 'from mllmcelltype', updated_content)
                updated_content = re.sub(r'~/.llmcelltype', '~/.mllmcelltype', updated_content)
                
                if content != updated_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Updated imports in {file_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python')
    
    update_imports(directory)
