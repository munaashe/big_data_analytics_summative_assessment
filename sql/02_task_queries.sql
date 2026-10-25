-- Task 2b: the three required statements, run against sql/01_schema.sql (SQLite 3).

-- investigation (a)
-- (a) INSERT a new casualty with valid references.
-- The example is fictional test data (collision 2024TEST000001) and is wrapped in a
-- transaction so that the collision, casualty and pedestrian detail are written atomically.
-- The leading DELETEs make the script repeatable.

DELETE FROM pedestrian_detail WHERE collision_index = '2024TEST000001';
DELETE FROM casualty          WHERE collision_index = '2024TEST000001';
DELETE FROM collision         WHERE collision_index = '2024TEST000001';

BEGIN TRANSACTION;
INSERT INTO collision (collision_index, collision_year, collision_ref_no)
VALUES ('2024TEST000001', 2024, 'TEST000001');

INSERT INTO casualty (collision_index, casualty_reference, vehicle_reference,
                      casualty_class_code, sex_code, age_of_casualty, severity_code,
                      enhanced_severity_code, injury_based, adjusted_serious_prob,
                      adjusted_slight_prob, casualty_type_code, car_passenger_code,
                      bus_passenger_code, imd_code, distance_code)
VALUES ('2024TEST000001', 1, 1,
        3,      -- Pedestrian
        2,      -- Female
        82,
        2,      -- Serious
        6,      -- Moderately Serious
        1,      -- injury-based reporting
        1.0, 0.0,
        0,      -- casualty_type: Pedestrian
        0,      -- Not car passenger
        0,      -- Not a bus or coach passenger
        4,      -- More deprived 30-40%
        1);     -- within 5km of home

INSERT INTO pedestrian_detail (collision_index, casualty_reference, ped_location_code,
                               ped_movement_code, road_worker_code)
VALUES ('2024TEST000001', 1,
        5,      -- In carriageway, crossing elsewhere
        1,      -- Crossing from driver's nearside
        0);     -- Not a road maintenance worker
COMMIT;

-- Verify the inserted row through the lookups.
SELECT c.collision_index, c.casualty_reference, cc.class_label, sx.sex_label,
       c.age_of_casualty, s.severity_label, pl.ped_location_label
FROM casualty c
JOIN casualty_class cc ON cc.class_code = c.casualty_class_code
JOIN severity s        ON s.severity_code = c.severity_code
LEFT JOIN sex sx       ON sx.sex_code = c.sex_code
LEFT JOIN pedestrian_detail pd ON pd.collision_index = c.collision_index
                              AND pd.casualty_reference = c.casualty_reference
LEFT JOIN pedestrian_location pl ON pl.ped_location_code = pd.ped_location_code
WHERE c.collision_index = '2024TEST000001';

-- investigation (b)
-- (b) FILTER: serious or fatal pedestrian casualties aged 70 or over who were in the
-- carriageway crossing away from a crossing facility (location 5) or within 50 m of one
-- (location 4).

-- Identifiers are deliberately not returned (data minimisation in displayed output).
SELECT s.severity_label, c.age_of_casualty,
       sx.sex_label, pl.ped_location_label, pm.ped_movement_label
FROM casualty c
JOIN pedestrian_detail pd   ON pd.collision_index = c.collision_index
                           AND pd.casualty_reference = c.casualty_reference
JOIN severity s             ON s.severity_code = c.severity_code
JOIN pedestrian_location pl ON pl.ped_location_code = pd.ped_location_code
LEFT JOIN pedestrian_movement pm ON pm.ped_movement_code = pd.ped_movement_code
LEFT JOIN sex sx            ON sx.sex_code = c.sex_code
WHERE c.casualty_class_code = 3
  AND c.severity_code IN (1, 2)
  AND c.age_of_casualty >= 70
  AND pd.ped_location_code IN (4, 5)
ORDER BY c.severity_code, c.age_of_casualty DESC, c.collision_index
LIMIT 10;

-- Row count for the same filter (reported alongside the sample rows).
SELECT COUNT(*) AS matching_casualties
FROM casualty c
JOIN pedestrian_detail pd ON pd.collision_index = c.collision_index
                         AND pd.casualty_reference = c.casualty_reference
WHERE c.casualty_class_code = 3
  AND c.severity_code IN (1, 2)
  AND c.age_of_casualty >= 70
  AND pd.ped_location_code IN (4, 5);

-- investigation (c)
-- (c) COUNT casualties by casualty type and severity (Fatal, Serious, Slight).
-- Conditional aggregation pivots the three severity categories into columns; casualties
-- with a missing type are shown as 'Missing'.

SELECT COALESCE(ct.casualty_type_label, 'Missing') AS casualty_type,
       SUM(CASE WHEN c.severity_code = 1 THEN 1 ELSE 0 END) AS fatal,
       SUM(CASE WHEN c.severity_code = 2 THEN 1 ELSE 0 END) AS serious,
       SUM(CASE WHEN c.severity_code = 3 THEN 1 ELSE 0 END) AS slight,
       COUNT(*) AS total
FROM casualty c
LEFT JOIN casualty_type ct ON ct.casualty_type_code = c.casualty_type_code
GROUP BY COALESCE(ct.casualty_type_label, 'Missing')
ORDER BY total DESC;
