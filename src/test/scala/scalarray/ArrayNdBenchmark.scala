package scalarray

import org.scalameter.api._

object ArrayNdBenchmark extends Bench.ForkedTime {

  private val numElements = Gen.range("elements")(500000, 1000000, 100000)

//  performance of "java array mapping" in {
//    val arrays = numElements.map(n => Array.fill[Int](n)(10))
//    measure method "map" in {
//      using(arrays) in { array =>
//        array.map(x => x + x)
//      }
//    }
//  }
//
//  performance of "ArrayNd underlying array mapping" in {
//    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
//    measure method "map" in {
//      using(arrays) in { array =>
//        array.elements.map(x => x + x)
//      }
//    }
//  }
//
//  performance of "ArrayNd untransposed mapping" in {
//    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
//    measure method "map" in {
//      using(arrays) in { array =>
//        array.map(x => x + x)
//      }
//    }
//  }
//
//  performance of "ArrayNd transposed mapping" in {
//    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10).reshape(n, 1).transpose)
//    measure method "map" in {
//      using(arrays) in { array =>
//        array.map(x => x + x)
//      }
//    }
//  }

  performance of "Broadcasting arrays of matching shape round 2" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    measure method "broadcast" in {
      using(arrays) in { array =>
        array.broadcast(array)(_ + _)
      }
    }
  }

}
