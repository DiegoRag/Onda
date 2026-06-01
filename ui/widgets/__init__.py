"""Reusable UI widgets that compose the views.

These are intentionally view-agnostic: a `TriplePlotCanvas` knows nothing
about the FFT Lab tab beyond "I draw three signals". Any tab that wants to
visualize multiple signals can reuse it.
"""

from ui.widgets.signal_state import SignalState
from ui.widgets.triple_plot_canvas import TriplePlotCanvas

__all__ = ["SignalState", "TriplePlotCanvas"]
