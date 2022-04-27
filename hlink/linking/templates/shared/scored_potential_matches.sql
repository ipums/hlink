{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT
pr.*
{% for c in pm_source_cols %}
	, pm.{{c}}
{% endfor %}
FROM {{predictions}} pr
JOIN {{potential_matches}} pm
ON pr.{{id_a}} = pm.{{id_a}} AND pr.{{id_b}} = pm.{{id_b}}
