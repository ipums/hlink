// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs

import org.apache.spark.sql.api.java.UDF2
import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset

class ExtractNeighbors extends UDF2[Seq[Row], Long, Seq[String]] {
  override def call(rows: Seq[Row], serial: Long): Seq[String] = {
    rows.filter(_.getLong(0) != serial).map(_.getString(1))
  }
}

