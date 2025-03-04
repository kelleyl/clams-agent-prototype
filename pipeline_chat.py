#!/usr/bin/env python3
"""
Command-line interface for the CLAMS Pipeline Generator.
This script provides an interactive chat interface for creating CLAMS pipelines.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.pipeline_generator import main

if __name__ == "__main__":
    main() 