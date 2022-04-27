// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

class ParseProbabilityVector extends UDF2[Vector, Int, Double] {
  override def call(v: Vector, i: Int): Double = {
    if (v.size < 2) {
        0
    } else {
    v(i)
    }
  }
}
