{# This file is part of the ISRDI's hlink. #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at: #}
{#   https://github.com/ipums/hlink #}

SELECT *
 , GREATEST(FLOAT(NAMEFRST_UNSTD_a_UNSTD_b_JW), FLOAT(NAMEFRST_UNSTD_a_STD_b_JW), FLOAT(NAMEFRST_STD_a_UNSTD_b_JW), FLOAT(NAMEFRST_STD_a_STD_b_JW)) as NAMEFRST_MAX_JW
 , ABS(AGE_a + 10 - AGE_b) as AGE_DIFF
FROM {{ plinks_accepted_ab_neighbors }}
