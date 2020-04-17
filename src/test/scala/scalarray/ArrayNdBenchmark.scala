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
    val indices = Gen.range("dimensionality")(0, 4, 1)
    val shapes = Seq(
      Seq(1000, 1),
      Seq(100, 10, 1, 1),
      Seq(10, 10, 10, 1, 1, 1),
      Seq(10, 10, 5, 2, 1, 1, 1, 1),
      Seq(10, 5, 2, 5, 2, 1, 1, 1, 1, 1)
    )
    val operands = indices.map {
      idx =>
        val shape = shapes(idx)
        val pair = (shape, shape.reverse)
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
