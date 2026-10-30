"""Bidding zones, weather points and global settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# Archived D-1 / D-2 weather forecasts (Open-Meteo previous-runs API) are complete from this date.
DATA_START = "2024-03-01"

# Day-ahead auction (SDAC) gate closure: 12:00 local time on D-1.
GATE_CLOSURE_HOUR = 12

ZONES = {
    "DE-LU": {
        "tz": "Europe/Berlin",
        "holidays": "DE",
        # (name, lat, lon): demand centres plus a northern wind cluster
        "points": [
            ("berlin", 52.52, 13.40),
            ("hamburg", 53.55, 10.00),
            ("munich", 48.14, 11.58),
            ("frankfurt", 50.11, 8.68),
            ("cologne", 50.94, 6.96),
            ("north_sea_coast", 54.30, 8.90),
        ],
    },
    "FR": {
        "tz": "Europe/Paris",
        "holidays": "FR",
        "points": [
            ("paris", 48.85, 2.35),
            ("lyon", 45.76, 4.84),
            ("marseille", 43.30, 5.37),
            ("lille", 50.63, 3.06),
            ("bordeaux", 44.84, -0.58),
            ("brittany", 48.11, -1.68),
        ],
    },
}

WEATHER_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
