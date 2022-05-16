// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.util.control.Breaks._
import scala.math.abs

class HHCompare extends UDF2[Seq[Row], Seq[Row], Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(y1: Seq[Row], y2: Seq[Row]): Double = {
    var score_tmp = Array[Double](0.0, 0.0, 0.0, 0.0, 0.0)
    var matches = 0.0
    for ((r2, i) <- y2.zipWithIndex) {
      breakable { for ((r1, j) <- y1.zipWithIndex) {
        score_tmp = Array(0.0, 0.0, 0.0, 0.0)
        score_tmp(0) = if (jw_sim.apply(r1.getAs[String]("namefrst_std"), r2.getAs[String]("namefrst_std")) > 0.8) 2.0 else 0.0
        score_tmp(1) = if (abs(r1.getAs[String]("birthyr").toLong - r2.getAs[String]("birthyr").toLong) < 1) 1.0 else 0.0
        score_tmp(2) = if (r1.getAs[String]("bpl") == r2.getAs[String]("bpl")) 1 else 0.0
        score_tmp(3) = if (r1.getAs[String]("sex") == r2.getAs[String]("sex")) 1 else 0.0
        if (score_tmp.sum > 3.9) {
          matches += 1
          break
        }
      } }
    }
    return matches / y2.length
  }
}
