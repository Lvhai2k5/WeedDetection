CREATE DATABASE IF NOT EXISTS weed_detection;
USE weed_detection;

CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    weed_density FLOAT,
    young_density FLOAT,
    mature_density FLOAT,
    weed_count INT,
    spray_time INT,
    blur_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
