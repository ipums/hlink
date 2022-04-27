{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT hh.serial, p_num.hh_name, p_num.neighborhood, row_number() OVER (PARTITION BY p_num.neighborhood ORDER BY p_num.serial) as num
FROM (
	SELECT {{sort_column}} as serial
	FROM prepped_df_tmp
	GROUP BY {{sort_column}} 
) hh
LEFT JOIN  (
	SELECT p.{{sort_column}} as serial, p.{{input_column}} as hh_name, p.{{neighborhood_column}} as neighborhood, row_number() OVER (PARTITION BY p.{{sort_column}} ORDER BY p.PERNUM) as num
	FROM prepped_df_tmp p
) p_num ON p_num.num = 1 AND p_num.serial = hh.serial

