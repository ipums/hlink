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

class AttachRelatedRows {
  def createAttachUDF(spark: SparkSession, df: Dataset[Row], transforms: java.util.List[java.util.Map[String, Any]], id_col: String, a_or_b: String, udf_name: String ) = {
    val attach_udf = (hh_rows: Seq[Row]) => {
      val related_rows_list = transforms.asScala.map { transform =>
        val input_cols = transform.get("input_cols").asInstanceOf[java.util.List[String]].asScala
        val filters = transform.get("filters").asInstanceOf[java.util.List[java.util.Map[String,Any]]].asScala
        val filtered_hh_rows = hh_rows.filter { row =>
          filters.map { filter => 
            val filter_col = filter.get("column").asInstanceOf[String]
            val min:Long = filter.get("min").asInstanceOf[Int].toLong
            val max:Long = filter.get("max").asInstanceOf[Int].toLong
            val filter_val = row.getAs[Long](filter_col).asInstanceOf[Int].toLong
            val has_dataset = filter.containsKey("dataset")
            if (!has_dataset || (has_dataset && filter.get("dataset").asInstanceOf[String] == a_or_b)) {
              (filter_val >= min.toLong) && (filter_val <= max.toLong)
              
            } else {
              true
            }
          }.forall(x => x)
        }
        filtered_hh_rows.map { row => (row.getAs[Any](id_col), Row.fromSeq(input_cols.map(row.getAs[Any](_)))) }
      }
      hh_rows.map { row =>
        val new_cols = related_rows_list.map { related_rows_list =>
          related_rows_list.filter { case (id, rel_row) =>
            val my_id = row.getAs[Any](id_col)
            my_id != id
          }.map(_._2)
        }
        Row.fromSeq(row.toSeq ++ new_cols)
      }
    }

    val old_struct_type = df.schema.find(_.name == "hh_rows").get.dataType.asInstanceOf[ArrayType].elementType.asInstanceOf[StructType]
    val old_struct_fields = old_struct_type.fields
    val new_struct_fields = transforms.asScala.map { transform =>
      val input_cols = transform.get("input_cols").asInstanceOf[java.util.List[String]].asScala
      val output_col = transform.get("output_col").asInstanceOf[String]
      val struct_fields = input_cols.map { input_col =>
        val data_type = old_struct_type.find(_.name == input_col).get.dataType
        StructField(input_col, data_type)
      }
      StructField(output_col, ArrayType(StructType(struct_fields)))
    }
    val schema = ArrayType(StructType(old_struct_fields ++ new_struct_fields))
    val my_udf = udf((hh_rows: Seq[Row]) => attach_udf(hh_rows), schema)
    spark.udf.register(udf_name, my_udf)
  }
}
