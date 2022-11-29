// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.commons.text.similarity._

class SerJaroWinklerSimilarity extends JaroWinklerSimilarity with Serializable {
  override def apply(left: CharSequence, right: CharSequence): java.lang.Double = {
    if (left == "" && right == "") {
      0.0
    } else {
      super.apply(left, right)
    }
  }
}
