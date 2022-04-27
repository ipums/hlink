{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT hh.serial, collect_list(hh2.hh_name) as neighbor_names
FROM hh_nbor_rank hh
LEFT JOIN hh_nbor_rank hh2 ON hh2.neighborhood = hh.neighborhood 
  AND hh2.serial != hh.serial 
	AND abs(hh2.num - hh.num) <= {{range}}
GROUP BY hh.serial
