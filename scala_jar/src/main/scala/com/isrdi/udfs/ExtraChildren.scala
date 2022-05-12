// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF8
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.util.control.Breaks._
import scala.collection.mutable.ArrayBuffer
import scala.math.abs

class ExtraChildren extends UDF8[Seq[Row], Seq[Row], String, Long, Long, String, String, Map[String, String], Double] {
  val jw_sim = new SerJaroWinklerSimilarity
  override def call(y1: Seq[Row], y2: Seq[Row], year_b: String, relate_a: Long, relate_b: Long, jw_threshold: String, age_threshold: String, var_map: Map[String, String]): Double = {

    if (relate_a <= 399 && relate_b <= 399) {
        var relate = var_map.getOrElse("relate", "relate")
        var year = year_b.toDouble
        var birthyr = var_map.getOrElse("byr", "birthyr")
        var CA = y1.filter(x => x.getAs[Long](relate) >= 300 && x.getAs[Long](relate) < 400)
        var CB = y2.filter(x => x.getAs[Long](relate) >= 300 && x.getAs[Long](relate) < 400 && (year - x.getAs[Long](birthyr) >= 11))

        if (CB.length > 0) {
            if (CA.length > 0) {
                var jw_t = jw_threshold.toDouble
                var age_t = age_threshold.toDouble
                var histid = var_map.getOrElse("histid", "histid")
                var name = var_map.getOrElse("name", "namefrst_std")
                var sex = var_map.getOrElse("sex", "sex")
                var ids_b = Set[String]()

                var good_matches = ArrayBuffer[Tuple3[Double, String, String]]()
                for ((r2, i) <- CB.zipWithIndex) {
                    for ((r1, j) <- CA.zipWithIndex) {
                        ids_b += r2.getAs[String](histid)
                        var jw_s = jw_sim.apply(r1.getAs[String](name), r2.getAs[String](name))
                        if (abs(r1.getAs[Long](birthyr).toLong - r2.getAs[Long](birthyr).toLong) <= age_t && r1.getAs[Long](sex) == r2.getAs[Long](sex) && jw_s >= jw_t) {
                            var ma = (jw_s.toDouble, r1.getAs[String](histid), r2.getAs[String](histid))
                            good_matches += ma
                        }
                    }
                }

                var tm_a = Set[String]()
                var tm_b = Set[String]()

                if (good_matches.nonEmpty) {
                    for (m <- good_matches.sortWith(_._1 > _._1)) {
                        if (!(tm_a contains m._2) && !(tm_b contains m._3)) {
                            tm_a += m._2
                            tm_b += m._3
                        }
                    }
                }
                var remaining_ids_b = ids_b &~ tm_b
                return remaining_ids_b.size
            } else {
                return CB.length
             }
        } else {
            return 0
        }
    } else {
        return 0
    }
  }
}
