{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT pd.*, pd_rel.{{output_col}}
FROM {{prepped_df}} pd
LEFT JOIN (
	SELECT pd.{{id}}, collect_set(
		if(pd_fam.{{input_cols[0]}} IS NOT NULL,
		named_struct(
		{% for c in input_cols %}
			'{{c}}', pd_fam.{{c}} {% if not loop.last %}, {% endif%}
		{% endfor %}
    ), NULL)
	) as {{output_col}}
	FROM {{prepped_df}} pd
	LEFT JOIN (
		SELECT *
		FROM {{prepped_df}} pd_fam
		WHERE pd_fam.{{relate_col}} <= {{top_code}} AND pd_fam.{{relate_col}} >= {{bottom_code}} 
	{% if filter %} AND {{filter}} {% endif %}
	) pd_fam ON pd_fam.{{family_id}} = pd.{{family_id}} AND pd_fam.{{id}} != pd.{{id}}
	GROUP BY pd.{{id}}
) pd_rel ON pd.{{id}} = pd_rel.{{id}}
