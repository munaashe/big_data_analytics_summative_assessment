"""Shared paths, code labels and helpers for the analysis pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "unprocessed"
RAW_CSV = RAW_DIR / "dft-road-casualty-statistics-casualty-2024.csv"
GUIDE_XLSX = RAW_DIR / "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx"

PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "casualty_2024_clean.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
SQL_DIR = ROOT / "sql"
SQL_EVIDENCE_DIR = ROOT / "outputs" / "sql_evidence"

RANDOM_STATE = 42

for d in (PROCESSED_DIR, FIG_DIR, TAB_DIR, SQL_EVIDENCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Labels taken from the 2024 DfT data guide (casualty table).
SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
CASUALTY_CLASS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
SEX = {1: "Male", 2: "Female", 9: "Unknown (self reported)", -1: "Missing"}
AGE_BAND = {1: "0-5", 2: "6-10", 3: "11-15", 4: "16-20", 5: "21-25", 6: "26-35",
            7: "36-45", 8: "46-55", 9: "56-65", 10: "66-75", 11: "Over 75", -1: "Missing"}
IMD = {1: "Most deprived 10%", 2: "More deprived 10-20%", 3: "More deprived 20-30%",
       4: "More deprived 30-40%", 5: "More deprived 40-50%", 6: "Less deprived 40-50%",
       7: "Less deprived 30-40%", 8: "Less deprived 20-30%", 9: "Less deprived 10-20%",
       10: "Least deprived 10%", -1: "Missing"}
DISTANCE = {1: "<=5km", 2: "5-10km", 3: "10-20km", 4: "20-100km", 5: ">100km", -1: "Missing"}
PED_LOCATION = {0: "Not a pedestrian", 1: "On pedestrian crossing", 2: "Zig-zag approach",
                3: "Zig-zag exit", 4: "Within 50m of crossing", 5: "In carriageway, crossing elsewhere",
                6: "On footway or verge", 7: "On refuge/island", 8: "Centre of carriageway",
                9: "In carriageway, not crossing", 10: "Unknown or other", -1: "Missing"}
PED_MOVEMENT = {0: "Not a pedestrian", 1: "Crossing from nearside", 2: "Nearside - masked",
                3: "Crossing from offside", 4: "Offside - masked", 5: "Stationary in carriageway",
                6: "Stationary - masked", 7: "Walking, facing traffic", 8: "Walking, back to traffic",
                9: "Unknown or other", -1: "Missing"}
CAR_PASSENGER = {0: "Not car passenger", 1: "Front seat", 2: "Rear seat", 9: "Unknown (self reported)",
                 -1: "Missing"}
BUS_PASSENGER = {0: "Not bus passenger", 1: "Boarding", 2: "Alighting", 3: "Standing", 4: "Seated",
                 9: "Unknown (self reported)", -1: "Missing"}
CASUALTY_TYPE = {0: "Pedestrian", 1: "Cyclist", 2: "Motorcycle <=50cc", 3: "Motorcycle 50-125cc",
                 4: "Motorcycle 125-500cc", 5: "Motorcycle >500cc", 8: "Taxi/Private hire",
                 9: "Car occupant", 10: "Minibus occupant", 11: "Bus or coach occupant",
                 16: "Horse rider", 17: "Agricultural vehicle", 18: "Tram occupant",
                 19: "Van/Goods <=3.5t", 20: "Goods 3.5-7.5t", 21: "Goods >=7.5t",
                 22: "Mobility scooter rider", 23: "Electric motorcycle", 90: "Other vehicle",
                 97: "Motorcycle unknown cc", 98: "Goods unknown weight",
                 99: "Unknown vehicle (self rep)", -1: "Missing"}

# Grouping of casualty_type into road-user groups used for Investigation B.
def road_user_group(code: int) -> str:
    if code == 0:
        return "Pedestrian"
    if code == 1:
        return "Cyclist"
    if code in (2, 3, 4, 5, 23, 97):
        return "Motorcyclist"
    if code in (8, 9):
        return "Car/taxi occupant"
    if code in (10, 11, 18):
        return "Bus/minibus/tram occupant"
    if code in (19, 20, 21, 98, 17):
        return "Goods/agricultural occupant"
    return "Other/unknown"


def savefig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
