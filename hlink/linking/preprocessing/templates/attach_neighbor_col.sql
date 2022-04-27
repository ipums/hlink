{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT p.*, hh.neighbor_names as {{output_column}}
FROM prepped_df_tmp p
LEFT JOIN hh_nbor hh ON p.{{sort_column}} = hh.serial
