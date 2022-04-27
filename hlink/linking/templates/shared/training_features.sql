{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT
{% if a_selects %}
{% for sel in a_selects %}
  a.{{sel}} as {{sel}}_a,
{% endfor %}
{% for sel in b_selects %}
	b.{{sel}} as {{sel}}_b,
{% endfor %}
{% else %}
a.{{id}} as {{id}}_a,
b.{{id}} as {{id}}_b,
{% endif %}
{{comp_features}},
{{match_feature}}

FROM training_data td
JOIN prepped_df_a a ON a.{{id}} = td.{{id}}_a
JOIN prepped_df_b b ON b.{{id}} = td.{{id}}_b

{% if distance_table %}
  {% for d in distance_table %}
    {{d}}
  {% endfor %}
{% endif %}
