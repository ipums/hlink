// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2
import scala.math.max

class MaxJWCompare extends UDF2[Seq[String], Seq[String], Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(list1: Seq[String], list2: Seq[String]): Double = {
    
    var max_score = 0.0
    for (s1 <- list1) {
      for (s2 <- list2) {
        max_score = max(max_score, jw_sim.apply(s1, s2))
      }
    }
    max_score
  }
}
