"""pytest bootstrap: make project `code/` importable as top-level packages
(utils, losses, models, eval, splits) without the stdlib `code` clash."""
import os
import sys

_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
