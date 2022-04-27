{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT
{% for c in id_columns %}
	{% if not loop.first %},{% endif %} {{c}} as {{c}}
{% endfor %}
{% for c in selected_columns %}
	, CAST({{c}} as FLOAT) as {{c}}
{% endfor %}
FROM {{df}}
