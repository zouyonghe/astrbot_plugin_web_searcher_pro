import os
import sys

PLUGIN_ROOT = os.path.dirname(os.path.dirname(__file__))
PLUGIN_PARENT = os.path.dirname(PLUGIN_ROOT)
for entry in (PLUGIN_PARENT,):
    if entry not in sys.path:
        sys.path.insert(0, entry)
