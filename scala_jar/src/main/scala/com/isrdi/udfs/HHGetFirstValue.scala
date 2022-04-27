// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs

import org.apache.spark.sql.api.java.UDF4
import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset

class HHRowsGetFirstValue extends UDF4[Seq[Row], String, String, String, Tuple2[Long, String]] {
  override def call(rows: Seq[Row], serial_col: String, pernum_col: String, value_col: String): Tuple2[Long, String] = {
    val min:Long = rows.map(_.getAs[Long](pernum_col)).min
    val row = rows.find(_.getAs[Long](pernum_col) == min).get
    (row.getAs[Long](serial_col), row.getAs[String](value_col))
  }
}
