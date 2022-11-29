// This file is part of the ISRDI's hlink.
// For copyright and licensing information, see the NOTICE and LICENSE files
// in this project's top-level directory, and also on-line at:
//   https://github.com/ipums/hlink

package com.isrdi.udfs
import org.apache.commons.text.similarity._

class SerJaroWinklerSimilarity extends JaroWinklerSimilarity with Serializable {
  // If either input string is empty, immediately return 0. Otherwise, return
  // the value of calling JaroWinklerSimilarity's apply function.
  // JaroWinklerSimilarity.apply("", "") returns 1, but for our use case it makes
  // more sense for two empty strings to have similarity 0.
  //
  // By my understanding, the comparison of two empty strings is essentially
  // undefined. To me it makes sense to directly apply the definition of
  // Jaro-Winkler similarity to two empty strings and get a similarity of 0.
  // But this has the unfortunate side effect of making the distance between two
  // empty strings 1 - 0 = 1. So I understand the decision to make the
  // similarity 1 and the distance 0 as well.
  override def apply(left: CharSequence, right: CharSequence): java.lang.Double = {
    if (left == "" || right == "") {
      0.0
    } else {
      super.apply(left, right)
    }
  }
}
