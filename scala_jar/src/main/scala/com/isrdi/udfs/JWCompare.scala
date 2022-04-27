// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2
import com.isrdi.udfs.SerJaroWinklerDistance


class JWCompare extends UDF2[String, String, Double] {
  val distance = new SerJaroWinklerDistance
  override def call(s1: String, s2: String): Double = {
    distance.apply(s1, s2)
  }
}
