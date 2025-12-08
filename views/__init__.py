"""
View registry for the VMEC dashboard.

Each feature lives in its own module with two simple entry points:
    - controls(): returns the left-panel layout for the feature.
    - render_*(): produces the figure for that feature.

To add a new feature (e.g. BoozerPlot), create a new module under views/
that exposes a controls() function plus a render function, then register it
in the VIEW_RENDERERS map inside test_app_fieldlines.py.
"""

from . import fieldline, overview, profiles, three_d, two_d

__all__ = ["fieldline", "overview", "profiles", "three_d", "two_d"]

