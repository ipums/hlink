{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT pd.*, pd_fam.{{other_col}} as {{output_col}}
FROM {{prepped_df}} pd
LEFT JOIN {{prepped_df}} pd_fam ON pd_fam.{{family_id}} = pd.{{family_id}} AND pd_fam.{{person_id}} = pd.{{person_pointer}}
