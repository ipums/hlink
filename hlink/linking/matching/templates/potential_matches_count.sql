{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT COUNT(1) 
FROM exploded_df_a a
JOIN exploded_df_b b ON 
  {% for col in blocking_columns %}
    a.{{ col }} = b.{{ col }} {{ "AND" if not loop.last }}
	{% endfor %}
