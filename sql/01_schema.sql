-- Task 2a: normalised (3NF) relational schema for the 2024 DfT casualty file.
-- Dialect: SQLite 3 (foreign keys enforced with PRAGMA foreign_keys = ON).
-- Codes and labels follow the 2024 DfT Road Safety Open Dataset data guide.
-- Missing values ("-1 Data missing or out of range") are stored as NULL.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS pedestrian_detail;
DROP TABLE IF EXISTS casualty;
DROP TABLE IF EXISTS collision;
DROP TABLE IF EXISTS severity;
DROP TABLE IF EXISTS enhanced_severity;
DROP TABLE IF EXISTS casualty_class;
DROP TABLE IF EXISTS sex;
DROP TABLE IF EXISTS age_band;
DROP TABLE IF EXISTS casualty_type;
DROP TABLE IF EXISTS imd_decile;
DROP TABLE IF EXISTS distance_band;
DROP TABLE IF EXISTS car_passenger_type;
DROP TABLE IF EXISTS bus_passenger_type;
DROP TABLE IF EXISTS pedestrian_location;
DROP TABLE IF EXISTS pedestrian_movement;
DROP TABLE IF EXISTS road_worker_status;

-- lookup tables
CREATE TABLE severity (
    severity_code   INTEGER PRIMARY KEY CHECK (severity_code IN (1, 2, 3)),
    severity_label  TEXT    NOT NULL UNIQUE
);
CREATE TABLE enhanced_severity (
    enhanced_code   INTEGER PRIMARY KEY,
    enhanced_label  TEXT    NOT NULL UNIQUE
);
CREATE TABLE casualty_class (
    class_code      INTEGER PRIMARY KEY,
    class_label     TEXT    NOT NULL UNIQUE
);
CREATE TABLE sex (
    sex_code        INTEGER PRIMARY KEY,
    sex_label       TEXT    NOT NULL UNIQUE
);
-- age_band is derived from age (age -> band); stored as a range lookup, not on casualty.
CREATE TABLE age_band (
    age_band_code   INTEGER PRIMARY KEY,
    age_band_label  TEXT    NOT NULL UNIQUE,
    min_age         INTEGER NOT NULL,
    max_age         INTEGER NOT NULL,
    CHECK (min_age <= max_age)
);
CREATE TABLE casualty_type (
    casualty_type_code  INTEGER PRIMARY KEY,
    casualty_type_label TEXT    NOT NULL,
    road_user_group     TEXT    NOT NULL
);
CREATE TABLE imd_decile (
    imd_code        INTEGER PRIMARY KEY CHECK (imd_code BETWEEN 1 AND 10),
    imd_label       TEXT    NOT NULL UNIQUE
);
CREATE TABLE distance_band (
    distance_code   INTEGER PRIMARY KEY,
    distance_label  TEXT    NOT NULL UNIQUE
);
CREATE TABLE car_passenger_type (
    car_passenger_code  INTEGER PRIMARY KEY,
    car_passenger_label TEXT    NOT NULL UNIQUE
);
CREATE TABLE bus_passenger_type (
    bus_passenger_code  INTEGER PRIMARY KEY,
    bus_passenger_label TEXT    NOT NULL UNIQUE
);
CREATE TABLE pedestrian_location (
    ped_location_code   INTEGER PRIMARY KEY,
    ped_location_label  TEXT    NOT NULL UNIQUE
);
CREATE TABLE pedestrian_movement (
    ped_movement_code   INTEGER PRIMARY KEY,
    ped_movement_label  TEXT    NOT NULL UNIQUE
);
CREATE TABLE road_worker_status (
    road_worker_code    INTEGER PRIMARY KEY,
    road_worker_label   TEXT    NOT NULL UNIQUE
);

-- core entities
-- Collision is a reference entity: only its key attributes are present in the supplied
-- casualty file (collision_index -> collision_year, collision_ref_no).
CREATE TABLE collision (
    collision_index   TEXT    PRIMARY KEY,
    collision_year    INTEGER NOT NULL CHECK (collision_year BETWEEN 1979 AND 2100),
    collision_ref_no  TEXT    NOT NULL,
    UNIQUE (collision_year, collision_ref_no)
);

CREATE TABLE casualty (
    collision_index        TEXT    NOT NULL REFERENCES collision (collision_index),
    casualty_reference     INTEGER NOT NULL CHECK (casualty_reference > 0),
    -- Source reference to a vehicle in the DfT vehicle table (not supplied); no FK.
    vehicle_reference      INTEGER NOT NULL CHECK (vehicle_reference > 0),
    casualty_class_code    INTEGER NOT NULL REFERENCES casualty_class (class_code),
    sex_code               INTEGER          REFERENCES sex (sex_code),
    age_of_casualty        INTEGER          CHECK (age_of_casualty BETWEEN 0 AND 120),
    severity_code          INTEGER NOT NULL REFERENCES severity (severity_code),
    enhanced_severity_code INTEGER          REFERENCES enhanced_severity (enhanced_code),
    injury_based           INTEGER NOT NULL CHECK (injury_based IN (0, 1)),
    adjusted_serious_prob  REAL    NOT NULL CHECK (adjusted_serious_prob BETWEEN 0 AND 1),
    adjusted_slight_prob   REAL    NOT NULL CHECK (adjusted_slight_prob BETWEEN 0 AND 1),
    casualty_type_code     INTEGER          REFERENCES casualty_type (casualty_type_code),
    car_passenger_code     INTEGER          REFERENCES car_passenger_type (car_passenger_code),
    bus_passenger_code     INTEGER          REFERENCES bus_passenger_type (bus_passenger_code),
    imd_code               INTEGER          REFERENCES imd_decile (imd_code),
    distance_code          INTEGER          REFERENCES distance_band (distance_code),
    PRIMARY KEY (collision_index, casualty_reference)
);

-- Optional 1:0..1 detail for pedestrian casualties only (casualty_class_code = 3).
CREATE TABLE pedestrian_detail (
    collision_index     TEXT    NOT NULL,
    casualty_reference  INTEGER NOT NULL,
    ped_location_code   INTEGER REFERENCES pedestrian_location (ped_location_code),
    ped_movement_code   INTEGER REFERENCES pedestrian_movement (ped_movement_code),
    road_worker_code    INTEGER REFERENCES road_worker_status (road_worker_code),
    PRIMARY KEY (collision_index, casualty_reference),
    FOREIGN KEY (collision_index, casualty_reference)
        REFERENCES casualty (collision_index, casualty_reference) ON DELETE CASCADE
);

-- Enforce that pedestrian details can only be attached to pedestrian casualties.
CREATE TRIGGER trg_pedestrian_only
BEFORE INSERT ON pedestrian_detail
WHEN (SELECT casualty_class_code FROM casualty
      WHERE collision_index = NEW.collision_index
        AND casualty_reference = NEW.casualty_reference) IS NOT 3
BEGIN
    SELECT RAISE(ABORT, 'pedestrian_detail requires a pedestrian casualty (class 3)');
END;

-- indexes
-- Supports filters/aggregations on severity and type (queries b and c).
CREATE INDEX idx_casualty_severity_type ON casualty (severity_code, casualty_type_code);
-- Supports class-based filtering (pedestrian queries).
CREATE INDEX idx_casualty_class ON casualty (casualty_class_code, severity_code);

-- analytical view
-- Denormalised read view for analysts; derived, not the source of record.
CREATE VIEW v_casualty_analysis AS
SELECT c.collision_index, c.casualty_reference, cl.collision_year,
       s.severity_label, cc.class_label, sx.sex_label, c.age_of_casualty,
       ab.age_band_label, ct.casualty_type_label, ct.road_user_group, im.imd_label
FROM casualty c
JOIN collision cl            ON cl.collision_index = c.collision_index
JOIN severity s              ON s.severity_code = c.severity_code
JOIN casualty_class cc       ON cc.class_code = c.casualty_class_code
LEFT JOIN sex sx             ON sx.sex_code = c.sex_code
LEFT JOIN age_band ab        ON c.age_of_casualty BETWEEN ab.min_age AND ab.max_age
LEFT JOIN casualty_type ct   ON ct.casualty_type_code = c.casualty_type_code
LEFT JOIN imd_decile im      ON im.imd_code = c.imd_code;
