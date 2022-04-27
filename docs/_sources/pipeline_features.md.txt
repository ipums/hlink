# Pipeline generated features 

## Transformer types

Each header below represents a feature created using a transformation available through the Spark Pipeline API.  These transforms are used in the context of `pipeline_features`.

```
[[pipeline_features]]
input_column = "immyear_diff"
output_column = "immyear_caution"
transformer_type = "bucketizer"
categorical = true
splits = [-1,0,6,11,9999]

[[pipeline_features]]
input_columns = ["race","srace"]
output_column = "race_interacted_srace"
transformer_type = "interaction"

```

### interaction

Interact two or more features, creating a vectorized result.

```
[[pipeline_features]]
# interact the categorical features for mother caution flag, mother present flag, and mother jaro-winkler score
input_columns = ["m_caution", "m_pres", "jw_m"]
output_column = "m_interacted_jw_m"
transformer_type = "interaction"
```

### bucketizer

From the `pyspark.ml.feature.Bucketizer()` docs: "Maps a column of continuous features to a column of feature buckets."

* Attributes:
  * `splits` -- Type: Array of integers.  Required for this transformer_type.  Per the `pyspark.ml.feature.Bucketizer()` docs: "Split points for mapping continuous features into buckets. With n+1 splits, there are n buckets. A bucket defined by splits x,y holds values in the range [x,y) except the last bucket, which also includes y. The splits should be of length >= 3 and strictly increasing. Values at -inf, inf must be explicitly provided to cover all Double values; otherwise, values outside the splits specified will be treated as errors."

```
[[pipeline_features]]
input_column = "relate_a"
output_column = "relatetype"
transformer_type = "bucketizer"
categorical = true
splits = [1,3,5,9999]
```
