{# This file is part of the ISRDI's hlink.                                   #}
{# For copyright and licensing information, see the NOTICE and LICENSE files #}
{# in this project's top-level directory, and also on-line at:               #}
{#   https://github.com/ipums/hlink                                    #}

SELECT *
FROM {{prepped_df}}
{% if universe_exprs %}
  WHERE 
  {% for expression in universe_exprs %}
      {{ expression }}
      {{ "AND" if not loop.last }}
  {% endfor %}
{% endif %}
