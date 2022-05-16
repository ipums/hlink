// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2


class JWCompare extends UDF2[String, String, Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(s1: String, s2: String): Double = {
    jw_sim.apply(s1, s2)
  }
}
