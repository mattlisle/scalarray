package scalarray

import org.scalameter.api._
import scala.reflect.ClassTag

object ArrayNdBenchmark extends Bench.ForkedTime {

  private val numElements = Gen.range("elements")(100000, 200000, 20000)

  performance of "java array mapping" in {
    val arrays = numElements.map(n => Array.fill[Int](n)(10))
    measure method "map" in {
      using(arrays) in { array =>
        array.map(x => x + x)
      }
    }
  }

  performance of "ArrayNd underlying array mapping" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    measure method "map" in {
      using(arrays) in { array =>
        array.elements.map(x => x + x)
      }
    }
  }

  performance of "ArrayNd mapping" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    measure method "map" in {
      using(arrays) in { array =>
        array.map(x => x + x)
      }
    }
  }

}
