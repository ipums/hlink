// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF3
import scala.math.max

class JWRate extends UDF3[Seq[String], Seq[String], String, Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(list1: Seq[String], list2: Seq[String], jw_threshold: String): Double = {
    
    var hits = 0.0
    var max_score = 0.0
    var jw_t = jw_threshold.toDouble
    for (s1 <- list1) {
      max_score = 0.0
      for (s2 <- list2) {
        max_score = max(max_score, jw_sim.apply(s1, s2))
      }
      if (max_score > jw_t) {
        hits = hits + 1.0
      }
    }
    if (list1.length > 0) {
        hits / list1.length
    } else{
        0
    }
  }
}
