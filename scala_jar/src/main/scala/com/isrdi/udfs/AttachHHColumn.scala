// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs

import org.apache.spark.sql.api.java.UDF5
import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import scala.util.control.Breaks._
import org.apache.spark.sql.SparkSession
import scala.math.abs
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConverters._

class AttachHHColumn {
  def createAttachUDF(spark: SparkSession, df: Dataset[Row], transforms: java.util.List[java.util.Map[String, String]], udf_name: String ) = {
    val person_id = transforms.get(0).get("person_id")
    val attach_udf = (hh_rows: Seq[Row]) => {
      val hh_map = hh_rows.map { row => row.getAs[Any](person_id) -> row }.toMap
      val new_rows = hh_rows.map { row =>
        val new_cols = transforms.asScala.map { transform =>
          val person_pointer = row.getAs[Any](transform.get("person_pointer"))
          if (person_pointer != 0) {
            val other_row = hh_map.get(person_pointer).get
            other_row.getAs[Any](transform.get("other_col"))
          } else {
            None
          }
        }
        Row.fromSeq(row.toSeq ++ new_cols)
      }
      new_rows
    }

    val old_struct_type = df.schema.find(_.name == "hh_rows").get.dataType.asInstanceOf[ArrayType].elementType.asInstanceOf[StructType]
    val old_struct_fields = old_struct_type.fields
    val new_struct_fields = transforms.asScala.map { transform =>
      val other_col = old_struct_type.find(_.name == transform.get("other_col")).get
      StructField(transform.get("output_col"), other_col.dataType)
    }
    val schema = ArrayType(StructType(old_struct_fields ++ new_struct_fields))
    val my_udf = udf((hh_rows: Seq[Row]) => attach_udf(hh_rows), schema)
    spark.udf.register(udf_name, my_udf)
  }
}
