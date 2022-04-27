{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT  /*+ BROADCAST(s) */ t.*
FROM {{table_name}} t
JOIN (
	SELECT DISTINCT t.serialp
	FROM {{table_name}} t
	JOIN training_data td ON t.{{id}} = td.{{id}}_{{a_or_b}}
) s ON s.serialp = t.serialp
