{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT pd.*, pd_rel.{{output_col}}
FROM {{prepped_df}} pd
LEFT JOIN (
	SELECT pd.{{id}}, collect_set(nvl(pd_fam.{{input_col}}, NULL)) as {{output_col}}
	FROM {{prepped_df}} pd
	LEFT JOIN {{prepped_df}} pd_fam ON pd_fam.{{family_id}} = pd.{{family_id}} AND pd_fam.{{id}} != pd.{{id}} AND pd_fam.{{relate_col}} <= {{top_code}} AND pd_fam.{{relate_col}} >= {{bottom_code}}
	GROUP BY pd.{{id}}
) pd_rel ON pd.{{id}} = pd_rel.{{id}}
