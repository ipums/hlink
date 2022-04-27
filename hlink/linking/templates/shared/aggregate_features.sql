{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT pm.*
  {% for feature in advanced_comp_features %}
    , agg.{{feature}}
  {% endfor %}
FROM {{ potential_matches }} pm
JOIN (
	SELECT 
	pm.{{id}}_a
	{% if "hits" in advanced_comp_features %}, COUNT(pm.{{id}}_b) as hits {% endif %}
	{% if "hits2" in advanced_comp_features %}, pow(COUNT(pm.{{id}}_b), 2) as hits2 {% endif %}
	{% if "exact_mult" in advanced_comp_features %}, SUM(CAST(pm.exact as INT)) > 1 as exact_mult {% endif %}
	{% if "exact_all_mult" in advanced_comp_features %}, SUM(CAST(pm.exact_all as INT)) as exact_all_mult {% endif %}
	{% if "exact_all_mult2" in advanced_comp_features %}, pow(SUM(CAST(pm.exact_all AS INT)), 2) as exact_all_mult2 {% endif %}
	FROM {{ potential_matches }} pm
	GROUP BY pm.{{id}}_a
) agg ON agg.{{id}}_a = pm.{{id}}_a

