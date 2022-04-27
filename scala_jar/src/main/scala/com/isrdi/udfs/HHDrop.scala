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
import scala.math.abs

class HHDrop extends UDF2[Seq[Row], String, Seq[Row]] {
  override def call(rows: Seq[Row], filter_id: String): Seq[Row] = {
    return rows.filterNot(_.getAs[String]("id") == filter_id)
  }
}
