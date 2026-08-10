# conftest.py — root level
# Adds project root to sys.path so pytest can import src/, simulation/ etc.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
