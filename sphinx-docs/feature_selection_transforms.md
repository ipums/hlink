# Feature Selection Transforms

Each feature selection in the `[[feature_selections]]` list must have a
`transform` attribute which tells hlink which transform it uses. The available
feature selection transforms are listed below. The attributes of the feature
selection often vary with the feature selection transform. However, there are a
few utility attributes which are available for all transforms:

- `override_column_a` - Type: `string`. Optional. Given the name of a column in
  dataset A, copy that column into the output column instead of computing the
  feature selection for dataset A. This does not affect dataset B.
- `override_column_b` - Type: `string`. Optional. Given the name of a column in
  dataset B, copy that column into the output column instead of computing the
  feature selection for dataset B. This does not affect dataset A.
- `set_value_column_a` - Type: any. Optional. Instead of computing the feature
  selection for dataset A, use the given value for every row in the output
  column. This does not affect dataset B.
- `set_value_column_b` - Type: any. Optional. Instead of computing the feature
  selection for dataset B, use the given value for every row in the output
  column. This does not affect dataset A.
- `checkpoint` - Type: `boolean`. Optional. If set to true, checkpoint the
  dataset in Spark before computing the feature selection. This can reduce some
  resource usage for very complex workflows, but should not be necessary.
- `skip` - Type: `boolean`. Optional. If set to true, don't compute this
  feature selection. This has the same effect as commenting the feature
  selection out of your config file.

## bigrams

Split the given string column into [bigrams](https://en.wikipedia.org/wiki/Bigram).

* Attributes:
  * `input_column` - Type: `string`. Required.
  * `output_column` - Type: `string`. Required.
  * `no_first_pad` - Type: boolean. Optional. If set to true, don't prepend a space " " to the column before splitting into bigrams. If false or not provided, do prepend the space.

```
[[feature_selections]]
input_column = "namelast_clean"
output_column = "namelast_clean_bigrams"
transform = "bigrams"
```

## sql_condition

Apply the given SQL.

* Attributes:
  * `condition` - Type: `string`. Required. The SQL condition to apply.
  * `output_column` - Type: `string`. Required.

```
[[feature_selections]]
input_column = "clean_birthyr"
output_column = "replaced_birthyr"
condition = "case when clean_birthyr is null or clean_birthyr == '' then year - age else clean_birthyr end"
transform = "sql_condition"
```

## array

Combine any number of input columns into a single array output column.

* Attributes:
  * `input_columns` - Type: list of strings. Required. The list of input columns.
  * `output_column` - Type: `string`. Required.

```
[[feature_selections]]
input_columns = ["namelast_clean_bigrams", "namefrst_unstd_bigrams"]
output_column = "namelast_frst_bigrams"
transform = "array"
```

## union

Take the set union of two columns that are arrays of strings, returning another
array of strings.

* Attributes:
  * `input_columns` - Type: list of strings. Required.
  * `output_column` - Type: `string`. Required.

## soundex

Compute the [soundex](https://en.wikipedia.org/wiki/Soundex) encoding of the input column.

* Attributes:
  * `input_column` - Type: `string`. Required.
  * `output_column` - Type: `string`. Required.

```
[[feature_selections]]
input_column = "namelast_clean"
output_column = "namelast_clean_soundex"
transform = "soundex"
```

## power

Raise the input column to a given power.

* Attributes:
  * `input_col` - Type: `string`. Required.
  * `output_col` - Type: `string`. Required.
  * `exponent` - Type: `int`. Required. The power to which to raise the input column.

```
[[feature_selections]]
input_col = "ncount"
output_col = "ncount2"
transform = "power"
exponent = 2
```

