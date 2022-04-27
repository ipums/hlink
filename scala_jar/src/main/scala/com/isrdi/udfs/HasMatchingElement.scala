// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.spark.sql.api.java.UDF2


class HasMatchingElement extends UDF2[String, String, Boolean] {
  override def call(l1: String, l2: String): Boolean = {
    return false;
    /*if (l1.size == 0 || l2.size == 0) {
      return false;
    }
    var it1 = l1.iterator;
    var it2 = l2.iterator;
    var cur1 = it1.next;
    var cur2 = it2.next;
    if (cur1 == cur2) {
      return true;
    }
    while(it1.hasNext && it2.hasNext) {
      if (cur1 == cur2) {
        return true;
      } else if (cur1 < cur2) {
        cur1 = it1.next;
      } else if (cur1 > cur2) {
        cur2 = it2.next;
      }
    }
    return false;*/
  }
}
