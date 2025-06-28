# Claude 4 Update Summary

## Date: June 28, 2025

This document summarizes all the changes made to reflect the Claude 4 release on June 27, 2025.

## Key Updates

### 1. Model Information
- Added Claude 4 Opus (claude-opus-4-20250514) 
- Added Claude 4 Sonnet (claude-sonnet-4-20250514)
- Kept existing Claude 3.7 and 3.5 models as they are still available
- Updated documentation to mention Claude 4 as the latest release

### 2. Files Updated

#### Python Package
- `/python/mllmcelltype/providers/anthropic.py`: Added Claude 4 model mappings
- `/python/mllmcelltype/functions.py`: Added Claude 4 models to supported models list and provider mapping
- `/python/README.md`: Updated to mention Claude 4 in provider list
- `/python/examples/scanpy_integration_example.py`: Updated example to use Claude 4 Opus

#### R Package
- `/R/R/get_provider.R`: 
  - Added Claude 4 models to anthropic_models list
  - Added Claude 4 models to OpenRouter models list
  - Updated documentation comments
- `/R/man/annotate_cell_types.Rd`: Updated model list documentation

#### Main Documentation
- `/README.md`: 
  - Updated introduction to mention Claude 4
  - Updated supported models section
  - Updated example code to use Claude 4 models
  - Updated consensus model recommendations to highlight Claude 4 as best
  - Changed "Claude-3.7/3.5" references to "Claude-4/3.7/3.5"

#### Translated READMEs
- `/README_CN.md`: Updated to mention Claude 4
- `/README_ES.md`: Updated to mention Claude 4
- `/README_JP.md`: Updated to mention Claude 4  
- `/README_DE.md`: Updated to mention Claude 4
- `/README_FR.md`: Updated to mention Claude 4
- `/README_KR.md`: Updated to mention Claude 4

### 3. Model Priority
- Claude 4 models are now listed first in Anthropic model lists
- Claude Opus 4 is recommended as the best consensus checking model
- Examples have been updated to use Claude 4 models by default

### 4. Backward Compatibility
- All existing Claude 3.7 and 3.5 models remain supported
- No breaking changes - existing code will continue to work
- Model mappings ensure smooth transition

## Note
The updates maintain full backward compatibility while adding support for the new Claude 4 models released on June 27, 2025.