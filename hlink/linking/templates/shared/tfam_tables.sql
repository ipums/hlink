{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT distinct
  {% for feature in cols %}
    {% if not loop.first %},{% endif %} rdf.{{feature}}
  {% endfor %}
FROM
(
  select distinct
  rdfs.serialp as serialp_{{a_or_b}}
  from
  {{ source_table}} hhpm
  JOIN
  raw_df_{{a_or_b}} rdfs
  ON
  hhpm.{{id}}_{{a_or_b}} = rdfs.{{id}}
) j
LEFT JOIN
raw_df_{{a_or_b}} rdf
ON rdf.serialp = j.serialp_{{a_or_b}}
