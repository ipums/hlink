{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT
{% if broadcast_hints %}
  {{broadcast_hints}}
{% endif %}
{% if broadcast_a_b %}
/*+ BROADCAST(a) */
/*+ BROADCAST(b) */
{% endif %}

pm.*

{% if comp_features %}
,{{comp_features}}
{% endif %}

FROM {{ potential_matches }} pm
JOIN prepped_df_a a ON a.{{id}} = pm.{{id}}_a
JOIN prepped_df_b b ON b.{{id}} = pm.{{id}}_b

{% if distance_table %}
  {% for d in distance_table %}
    {{d}}
  {% endfor %}
{% endif %}
