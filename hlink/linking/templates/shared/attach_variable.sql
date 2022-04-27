{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT p.*, rd.{{col_to_add}} as {{output_col}}
FROM {{prepped_df}} p
LEFT JOIN {{region_data}} rd ON rd.{{col_to_join_on}} = p.{{input_col}}
