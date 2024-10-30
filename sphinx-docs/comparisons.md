# Comparisons

## Overview

The `comparisons` configuration section defines constraints on the matching
process. Unlike `comparison_features` and `feature_selections`, which define
features for use with a machine-learning algorithm, `comparisons` define rules
which directly filter the output `potential_matches` table. These rules often
depend on some comparison features, and hlink always applies the rules after
exploding and blocking in the matching task.

As an example, suppose that your `comparisons` configuration section looks like
the following.

```
[comparisons]
comparison_type = "threshold"
feature_name = "namefrst_jw"
threshold = 0.79
```

This comparison defines a rule that depends on the `namefrst_jw` comparison
feature. During matching, only pairs of records with `namefrst_jw` greater than
or equal to 0.79 will be added to the potential matches table. Pairs of records
which do not satisfy the comparison will not be potential matches.

*Note: This page focuses on the `comparisons` section in particular, but the
household comparisons section `hh_comparisons` has the same structure. It
defines rules which hlink uses to filter record pairs after household blocking
in the hh_matching task. These rules are effectively filters on the output
`hh_potential_matches` table.*

## Comparison Types

Currently the only `comparison_type` supported for the `comparisons` section is
`"threshold"`. This requires the `threshold` attribute, and by default, it
restricts a comparison feature to be greater than or equal to the value given
by `threshold`. The configuration section

```
[comparisons]
comparison_type = "threshold"
feature_name = "namelast_jw"
threshold = 0.84
```

adds the condition `namelast_jw >= 0.84` to each record pair considered during
matching. Only record pairs which satisfy this condition are marked as
potential matches.

Hlink also supports a `threshold_expr` attribute in `comparisons` for more
flexibility. This attribute takes SQL syntax and replaces the `threshold`
attribute described above. For example, to define the condition `flag < 0.5`,
you could set `threshold_expr` like

```
[comparisons]
comparison_type = "threshold"
feature_name = "flag"
threshold_expr = "< 0.5"
```

Note that there is now no need for the `threshold` attribute because the
`threshold_expr` implicitly defines it.

## Defining Multiple Comparisons

In some cases, you may have multiple comparisons to make between record pairs.
The `comparisons` section supports this in a flexible but somewhat verbose way.
Suppose that you would like to combine two of the conditions used in the
examples above, so that record pairs are potential matches only if `namefrst_jw >= 0.79`
and `namelast_jw >= 0.84`. You could do this by setting the `operator`
attribute to `"AND"` and then defining the `comp_a` (comparison A) and `comp_b`
(comparison B) attributes.

```
[comparisons]
operator = "AND"

[comparisons.comp_a]
comparison_type = "threshold"
feature_name = "namefrst_jw"
threshold = 0.79

[comparisons.comp_b]
comparison_type = "threshold"
feature_name = "namelast_jw"
threshold = 0.84
```

Both `comp_a` and `comp_b` are recursive, so they may have the same structure
as the `comparisons` section itself. This means that you can add as many
comparisons as you would like by recursively defining comparisons. `operator`
may be either `"AND"` or `"OR"` and defines the logic for connecting the two
sub-comparisons `comp_a` and `comp_b`. Defining more than two comparisons can
get pretty ugly and verbose, so make sure to use care when defining nested
comparisons. Here is an example of a section with three comparisons.

```
# This comparisons section defines 3 rules for potential matches.
# They are that potential matches must either have
# 1. flag < 0.5
# OR
# 2. namefrst_jw >= 0.79 AND 3. namelast_jw >= 0.84
[comparisons]
operator = "OR"

[comparisons.comp_a]
comparison_type = "threshold"
feature_name = "flag"
threshold_expr = "< 0.5"

[comparisons.comp_b]
operator = "AND"

[comparisons.comp_b.comp_a]
comparison_type = "threshold"
feature_name = "namefrst_jw"
threshold = 0.79

[comparisons.comp_b.comp_b]
comparison_type = "threshold"
feature_name = "namelast_jw"
threshold = 0.84
```
