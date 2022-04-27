{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT pm.*
  {% if "jw_max_a" in hh_comp_features %}
  , coalesce(
    greatest(
      max(case when pm.byrdiff <= 10 then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b ORDER BY pm.{{id}}_a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
      max(case when pm.byrdiff <= 10 then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b ORDER BY pm.{{id}}_a ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
    ),
    max(case when pm.byrdiff <= 10 then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b ORDER BY pm.{{id}}_a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    max(case when pm.byrdiff <= 10 then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b ORDER BY pm.{{id}}_a ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING),
    0
    ) as jw_max_a {% endif %}
	{% if "jw_max_b" in hh_comp_features %}
	, coalesce(
      greatest(
        max(case when pm.byrdiff <= 10 and pm.sexmatch then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b order by pm.{{id}}_a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        max(case when pm.byrdiff <= 10 and pm.sexmatch then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b order by pm.{{id}}_a ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
      ),
      max(case when pm.byrdiff <= 10 and pm.sexmatch then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b order by pm.{{id}}_a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
      max(case when pm.byrdiff <= 10 and pm.sexmatch then pm.namefrst_jw else 0 end) over(PARTITION BY pm.{{id}}_b, pm.{{hh_col}}_a, pm.{{hh_col}}_b order by pm.{{id}}_a ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING),
      0
	  ) as jw_max_b {% endif %}
FROM {{potential_matches}} pm
