package scalarray

import org.scalameter.api._

object ArrayNdBenchmark extends Bench.ForkedTime {

  private val numElements = Gen.range("elements")(500000, 1000000, 100000)

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

  performance of "ArrayNd contiguous mapping" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    measure method "map" in {
      using(arrays) in { array =>
        array.map(x => x + x)
      }
    }
  }

  performance of "ArrayNd non-contiguous mapping" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10).reshape(n, 1).transpose)
    measure method "map" in {
      using(arrays) in { array =>
        array.map(x => x + x)
      }
    }
  }

  performance of "Broadcasting arrays of matching shape round 2" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    val toAdd = arrays.map(array => array.broadcastWith(array)(_))
    measure method "broadcast" in {
      using(toAdd) in { a =>
        a(_ + _)
      }
    }
  }

  performance of "broadcasted array-array multiplication" in {
    val numElements = Gen.range("elements")(500, 1000, 100)
    val shapes = numElements.map { n =>
      val x = n / 10
      val y = 10
      (Seq(x, y, 1, 1), Seq(1, 1, x, y))
    }
    val operands = shapes.map {
      pair =>
        val arr = ArrayNd.fill[Int](pair._1.product)(1)
        (arr.reshape(pair._1: _*), arr.reshape(pair._2: _*))
    }
    val toAdd = operands.map(ops => ops._1.broadcastWith(ops._2)(_))
    measure method "broadcast" in {
      using(toAdd) in { a =>
        a(_ + _)
      }
    }
  }

}
