{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT DISTINCT 
{% for c in dataset_columns %}
	{% if not loop.first %},{% endif %}a.{{c}} as {{c}}_a
	,b.{{c}} as {{c}}_b
{% endfor %}
{% if feature_columns %}
  {% for c in feature_columns %}
    ,{{c}}
  {% endfor %}
{% endif %}
FROM exploded_df_a a
JOIN exploded_df_b b ON 
{% for col in blocking_columns %}
a.{{ col }} = b.{{ col }} {{ "AND" if not loop.last }}
{% endfor %}
{% if distance_table %}
  {% for d in distance_table %}
    {{d}}
  {% endfor %}
{% endif %}
{% if matching_clause %}
WHERE 
{{ matching_clause }}
{% endif %}
