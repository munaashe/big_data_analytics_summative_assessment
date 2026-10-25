"""Draw the Crow's Foot ERD (Figure 8) and the scalable architecture diagram (Figure 9)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from config import savefig

ROW_H = 1.9


class Entity:
    def __init__(self, ax, name, x, y, w, rows, color="#dbe9f6"):
        self.x, self.w = x, w
        self.h = ROW_H * (len(rows) + 1)
        self.y = y  # top edge
        ax.add_patch(Rectangle((x, y - ROW_H), w, ROW_H, fc="#2c3e50", ec="black", lw=0.8))
        ax.text(x + w / 2, y - ROW_H / 2, name, ha="center", va="center", color="white",
                fontsize=7.5, weight="bold")
        ax.add_patch(Rectangle((x, y - self.h), w, self.h - ROW_H, fc=color, ec="black", lw=0.8))
        for i, (key, text) in enumerate(rows):
            yy = y - ROW_H * (i + 1.5)
            ax.text(x + 0.4, yy, key, fontsize=5.6, va="center", weight="bold", color="#8e2c1c")
            ax.text(x + 4.6, yy, text, fontsize=5.6, va="center",
                    style="italic" if "(derived" in text else "normal")

    @property
    def center(self):
        return np.array([self.x + self.w / 2, self.y - self.h / 2])

    def edge_point(self, toward):
        c = self.center
        d = np.asarray(toward, float) - c
        sx = (self.w / 2) / abs(d[0]) if d[0] else np.inf
        sy = (self.h / 2) / abs(d[1]) if d[1] else np.inf
        return c + d * min(sx, sy)


def crow_end(ax, p, u, kind):
    """Draw a Crow's Foot end marker at point p on an entity, u = unit vector away from it."""
    perp = np.array([-u[1], u[0]])
    L, W = 1.6, 0.9
    if "many" in kind:
        tip = p + u * L
        for s in (-1, 0, 1):
            ax.plot(*zip(tip, p + perp * W * s), color="black", lw=0.8)
    bar1 = p + u * (L + 0.5)
    ax.plot(*zip(bar1 + perp * W, bar1 - perp * W), color="black", lw=0.8) if kind in ("one", "one_many") else None
    if kind == "one":
        bar0 = p + u * 0.6
        ax.plot(*zip(bar0 + perp * W, bar0 - perp * W), color="black", lw=0.8)
    if kind in ("zero_one", "zero_many"):
        bar = p + u * (0.6 if kind == "zero_one" else L + 0.4)
        if kind == "zero_one":
            ax.plot(*zip(bar + perp * W, bar - perp * W), color="black", lw=0.8)
        circ = p + u * (L + 1.3)
        ax.add_patch(plt.Circle(circ, 0.6, fc="white", ec="black", lw=0.8, zorder=5))


def relate(ax, a, b, end_a, end_b, label=None, dashed=False):
    pa, pb = a.edge_point(b.center), b.edge_point(a.center)
    ax.plot(*zip(pa, pb), color="black", lw=0.8, ls="--" if dashed else "-", zorder=1)
    u = (pb - pa) / np.linalg.norm(pb - pa)
    if end_a:
        crow_end(ax, pa, u, end_a)
    if end_b:
        crow_end(ax, pb, -u, end_b)
    if label:
        m = (pa + pb) / 2
        ax.text(*m, label, fontsize=5.2, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.3))


def erd():
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 82)
    ax.axis("off")

    cas = Entity(ax, "CASUALTY", 44, 64, 30, [
        ("PK,FK", "collision_index  TEXT"), ("PK", "casualty_reference  INT"),
        ("", "vehicle_reference  INT (source ref.)"), ("FK", "casualty_class_code  NOT NULL"),
        ("FK", "sex_code  NULL"), ("", "age_of_casualty  INT NULL, 0-120"),
        ("FK", "severity_code  NOT NULL"), ("FK", "enhanced_severity_code  NULL"),
        ("", "injury_based  0/1"), ("", "adjusted_serious_prob  REAL"),
        ("", "adjusted_slight_prob  REAL"), ("FK", "casualty_type_code  NULL"),
        ("FK", "car_passenger_code  NULL"), ("FK", "bus_passenger_code  NULL"),
        ("FK", "imd_code  NULL"), ("FK", "distance_code  NULL")], color="#fdebd0")
    col = Entity(ax, "COLLISION (reference)", 4, 64, 26, [
        ("PK", "collision_index  TEXT"), ("", "collision_year  INT"),
        ("", "collision_ref_no  TEXT"), ("UQ", "(collision_year, collision_ref_no)")], color="#fdebd0")
    ped = Entity(ax, "PEDESTRIAN_DETAIL", 82, 64, 27, [
        ("PK,FK", "collision_index"), ("PK,FK", "casualty_reference"),
        ("FK", "ped_location_code  NULL"), ("FK", "ped_movement_code  NULL"),
        ("FK", "road_worker_code  NULL")], color="#fdebd0")

    def lk(name, x, y, w=21):
        return Entity(ax, name, x, y, w, [("PK", "code  INT"), ("", "label  TEXT")])

    sev = lk("SEVERITY", 30, 80, 17)
    cls = lk("CASUALTY_CLASS", 50, 80, 19)
    sex = lk("SEX", 72, 80, 14)
    enh = lk("ENHANCED_SEVERITY", 4, 46)
    typ = Entity(ax, "CASUALTY_TYPE", 4, 34, 24, [("PK", "code  INT"), ("", "label  TEXT"),
                                                  ("", "road_user_group  TEXT")])
    age = Entity(ax, "AGE_BAND", 4, 18, 22, [("PK", "code  INT"), ("", "label  TEXT"),
                                             ("", "min_age, max_age  INT")])
    imd = lk("IMD_DECILE", 32, 14, 18)
    dist = lk("DISTANCE_BAND", 52, 14, 19)
    car = lk("CAR_PASSENGER_TYPE", 73, 14)
    bus = lk("BUS_PASSENGER_TYPE", 84, 30)
    loc = lk("PEDESTRIAN_LOCATION", 118, 80)
    mov = lk("PEDESTRIAN_MOVEMENT", 118, 62)
    rw = lk("ROAD_WORKER_STATUS", 118, 42)

    relate(ax, col, cas, "one", "one_many", "records")
    relate(ax, cas, ped, "one", "zero_one")
    for e in (sev, cls):
        relate(ax, e, cas, "one", "zero_many")
    for e in (sex, enh, typ, imd, dist, car, bus):
        relate(ax, e, cas, "zero_one", "zero_many")
    for e in (loc, mov, rw):
        relate(ax, e, ped, "zero_one", "zero_many")
    relate(ax, age, cas, None, None, dashed=True)
    ax.text(20, 22, "derived: age BETWEEN min_age\nAND max_age (view, no FK)", fontsize=5.5)

    ax.text(70, 1.5, "Crow's Foot notation: || one and only one;  o| zero or one;  |< one or many;  o< zero or "
                     "many.  Dashed = derived (no FK).  Vehicle table not supplied: vehicle_reference kept as a "
                     "source reference without FK.", ha="center", fontsize=6.5)
    ax.set_title("Normalised (3NF) ERD for the 2024 casualty data (SQLite implementation)", fontsize=10)
    savefig(fig, "fig08_erd.png")
    plt.close(fig)


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec="#2c3e50", lw=0.9))
    ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(x + w / 2, y + h - 3.6, body, ha="center", va="top", fontsize=6.0, linespacing=1.3)


def arrow(ax, p, q, dashed=False, text=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                 ls="--" if dashed else "-", color="#2c3e50"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.8, text, ha="center", fontsize=5.5)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 62)
    ax.axis("off")
    batch, stream, serve = "#d6eaf8", "#fdebd0", "#d5f5e3"

    ax.text(1, 59, "CURRENT: annual batch path", fontsize=8, weight="bold", color="#1f4e79")
    box(ax, 1, 38, 16, 18, "1. Source", "DfT annual casualty\nCSV release (+ guide)\nmonthly/annual\nrevisions", batch)
    box(ax, 21, 38, 18, 18, "2. Landing (raw)", "Object storage\n(e.g. S3/ADLS)\nimmutable, versioned\nchecksum + lineage\nyear=YYYY/ prefix", batch)
    box(ax, 43, 38, 18, 18, "3. Validate", "Schema + code-list\nchecks vs data guide\nkey uniqueness, -1->NULL\nquarantine failures", batch)
    box(ax, 65, 38, 20, 18, "4. Process (Spark)", "Distributed cluster\ncurated Parquet/Delta\npartitioned by year\n(+police force / region)\naggregates & features", batch)
    box(ax, 89, 38, 15, 18, "5. Serve", "Warehouse / SQL\nengine (3NF-derived\nstar & aggregates)\nRDBMS source of\nrecord", serve)
    box(ax, 107, 47, 12.5, 9, "Dashboards", "refresh after\nrelease", serve)
    box(ax, 107, 36, 12.5, 9, "Alerts", "fatal-count\nrules engine", serve)
    for a, b in [((17.6, 47), (20.4, 47)), ((39.6, 47), (42.4, 47)), ((61.6, 47), (64.4, 47)),
                 ((85.6, 47), (88.4, 47)), ((104.6, 50), (106.4, 51)), ((104.6, 44), (106.4, 40))]:
        arrow(ax, a, b)

    ax.text(1, 31, "FUTURE (only if near-miss / telematics feed is approved)", fontsize=8, weight="bold",
            color="#a04000")
    box(ax, 1, 12, 16, 16, "A. Event source", "Telematics / app\nnear-miss events\n(consented,\npseudonymised)", stream)
    box(ax, 21, 12, 18, 16, "B. Durable log", "Kafka topics\npartition key = region\nretention for replay\nschema registry", stream)
    box(ax, 43, 12, 18, 16, "C. Stream processing", "Spark Structured\nStreaming / Flink\nevent-time windows,\nwatermarks, dedup\n(idempotent sink)", stream)
    box(ax, 65, 12, 20, 16, "D. Validated stream\ntables", "\nsame curated store\nreconciled with batch\nwhen official data\narrives", stream)
    for a, b in [((17.6, 20), (20.4, 20)), ((39.6, 20), (42.4, 20)), ((61.6, 20), (64.4, 20))]:
        arrow(ax, a, b, dashed=True)
    arrow(ax, (85.6, 22), (106.4, 38), dashed=True, text="near-real-time alerts")
    arrow(ax, (75, 28.6), (75, 37.4), dashed=True, text="")

    ax.add_patch(FancyBboxPatch((1, 1), 118, 7, boxstyle="round,pad=0.3", fc="#eaeded", ec="#2c3e50", lw=0.9))
    ax.text(60, 4.5, "Cross-cutting: orchestration (e.g. Airflow DAG per release)  |  data-quality & pipeline monitoring  |  "
                     "access control, encryption, audit log  |  lineage/catalogue  |  backup & replay from raw",
            ha="center", va="center", fontsize=6.6)
    ax.set_title("Proposed architecture: annual batch path (solid) and optional future stream path (dashed)",
                 fontsize=10)
    savefig(fig, "fig09_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    erd()
    architecture()
