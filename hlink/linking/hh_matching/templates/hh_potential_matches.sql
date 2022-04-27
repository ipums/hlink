{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT DISTINCT 
hhbm.*
{% if feature_columns %}
  {% for c in feature_columns %}
    , {{c}}
  {% endfor %}
{% endif %}

FROM hh_blocked_matches hhbm
JOIN prepped_df_a a

JOIN prepped_df_b b
ON
a.{{id_col}} == hhbm.{{id_col}}_a
AND
b.{{id_col}} == hhbm.{{id_col}}_b

{% if matching_clause %}
WHERE 
{{ matching_clause }}
{% endif %}
