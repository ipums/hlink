// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.ml.linalg.Vector

class VectorToString extends UDF1[Vector, String] {
  def stringify(x: Vector): String = x match {
    case null => null
    case _ => x.toString
  }
  override def call(s1: Vector): String = {
    stringify(s1)
  }
}
