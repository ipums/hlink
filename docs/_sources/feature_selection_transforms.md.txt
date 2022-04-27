# Feature Selection transforms

Each header below represents a feature selection transform.  These transforms are used in the context of `feature_selections`.

```
[[feature_selections]]
input_column = "clean_birthyr"
output_column = "replaced_birthyr"
condition = "case when clean_birthyr is null or clean_birthyr == '' then year - age else clean_birthyr end"
transform = "sql_condition"
```

There are some additional attributes available for all transforms: `checkpoint`, `override_column_a`, `override_column_b`, `set_value_column_a`, `set_value_column_b`.

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

Combine two input columns into an array output column.

* Attributes:
  * `input_columns` - Type: list of strings. Required. The two input columns.
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

