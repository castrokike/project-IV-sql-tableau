ALTER TABLE `friends`.`seasons` 
CHANGE COLUMN `ep_number_overall` `ep_number_overall` INT NULL DEFAULT NULL ,
CHANGE COLUMN `ep_number_season` `ep_number_season` INT NULL DEFAULT NULL ,
CHANGE COLUMN `us_viewers_mm` `us_viewers_mm` DECIMAL(3) NULL DEFAULT NULL ,
CHANGE COLUMN `rating_1` `rating_1` DECIMAL(3) NULL DEFAULT NULL ,
CHANGE COLUMN `rating_2` `rating_2` DECIMAL(3) NULL DEFAULT NULL ,
CHANGE COLUMN `us_viewers_mm_2` `us_viewers_mm_2` DECIMAL(3) NULL DEFAULT NULL ,
CHANGE COLUMN `season` `season` INT NULL DEFAULT NULL ;

