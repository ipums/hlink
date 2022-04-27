{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT 
{% for sel in a_selects %}
  pa.{{sel}} as {{sel}}_a,
{% endfor %}
{% for sel in b_selects %}
	pb.{{sel}} as {{sel}}_b{% if not(loop.last) %},{% endif %}
{% endfor %}

FROM training_data td
JOIN prepped_df_a pa ON pa.id = td.id_a
JOIN prepped_df_b pb ON pb.id = td.id_b
