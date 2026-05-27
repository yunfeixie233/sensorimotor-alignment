"""Google-brand 4-color palette + light→dark linear colormaps for ranked
color intensity within a panel. Imported by every figure-generator script."""

# Resolve paths relative to this script's location so the release works from any clone path.
import os as _os
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[3]
_RECORD_FINAL = _REPO_ROOT / "record" / "1_final"
_FIGURES = _REPO_ROOT / "record" / "1_final" / "figures"

from matplotlib.colors import LinearSegmentedColormap

GOOGLE_BLUE   = "#4285F4"
GOOGLE_RED    = "#EA4335"
GOOGLE_YELLOW = "#FBBC04"
GOOGLE_GREEN  = "#34A853"

# Light → saturated linear ramps so darker = higher value within a panel.
GOOGLE_CMAP = {
    "blue":   LinearSegmentedColormap.from_list("g_blue",   ["#D2E3FC", GOOGLE_BLUE]),
    "red":    LinearSegmentedColormap.from_list("g_red",    ["#FAD2CF", GOOGLE_RED]),
    "yellow": LinearSegmentedColormap.from_list("g_yellow", ["#FEEFC3", GOOGLE_YELLOW]),
    "green":  LinearSegmentedColormap.from_list("g_green",  ["#CEEAD6", GOOGLE_GREEN]),
    "grey":   LinearSegmentedColormap.from_list("g_grey",   ["#E8EAED", "#5F6368"]),
}
