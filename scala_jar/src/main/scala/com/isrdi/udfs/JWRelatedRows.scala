// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF5
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.util.control.Breaks._
import scala.math.abs

class JWRelatedRows extends UDF5[Seq[Row], Seq[Row], String, String, Map[String, String], Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(y1: Seq[Row], y2: Seq[Row], jw_threshold: String, age_threshold: String, var_map: Map[String, String]): Double = {
    var score_tmp = Array[Double](0.0, 0.0, 0.0, 0.0)
    var matches = 0.0
    var jw_t = jw_threshold.toDouble
    var age_t = age_threshold.toDouble
    var name = var_map.getOrElse("name", "namefrst_std")
    var byr = var_map.getOrElse("byr", "birthyr")
    var sex = var_map.getOrElse("sex", "sex")
    for ((r2, i) <- y2.zipWithIndex) {
      breakable { for ((r1, j) <- y1.zipWithIndex) {
        score_tmp = Array(0.0, 0.0, 0.0)
        score_tmp(0) = if (jw_sim.apply(r1.getAs[String](name), r2.getAs[String](name)) >= jw_t) 1.0 else 0.0
        score_tmp(1) = if (abs(r1.getAs[Long](byr).toLong - r2.getAs[Long](byr).toLong) <= age_t) 1.0 else 0.0
        score_tmp(2) = if (r1.getAs[Long](sex) == r2.getAs[Long](sex)) 1 else 0.0
        if (score_tmp.sum == 3) {
          matches += 1
          break
        }
      } }
    }
    return matches
  }
}
