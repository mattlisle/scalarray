package scalarray

import org.scalameter.api._
import scala.reflect.ClassTag

object ArrayNdBenchmark extends Bench.ForkedTime {

  private val numElements = Gen.range("elements")(10000, 50000, 10000)

  case class BroadcastedArray(a: Array[Int], b: Array[Int]) {
    require(a.length == b.length)
    private val length = a.length
    def broadcast(f: (Int, Int) => Int): Array[Int] = {
      val c = new Array[Int](length)
      var idx = 0
      while (idx < length) {
        c(idx) = f(a(idx), b(idx))
        idx += 1
      }
      c
    }
  }

  case class BroadcastedArrayNd[@specialized(Int, Double) A](a: ArrayNd[A], b: ArrayNd[A]) {
    require(a.length == b.length)
    private val length = a.length
    def broadcast(f: (A, A) => A)(implicit numeric: Numeric[A], classTag: ClassTag[A]): Unit = {
      val c = new Array[A](length)
      var idx = 0
      while (idx < length) {
        c(idx) = f(a.elements(idx), b.elements(idx))
        idx += 1
      }
//      val shp = Seq(length)
//      new ArrayNd(c, Seq(length), false)
    }
  }

  performance of "same-shape array multiplication using built-in iterator" in {
    val operands = numElements.map(n => (ArrayNd.fill[Int](n)(10), ArrayNd.fill[Int](n)(5)))
    measure method "+" in {
      using(operands) in { operand =>
        operand._1 + operand._2
      }
    }
  }

  performance of "one-dimensional array multiplication" in {
    val arrays = numElements.map(n => Array.fill[Int](n)(10))
    val broadcasts = arrays.map(arr => BroadcastedArray(arr, arr))
    measure method "evaluate" in {
      using(broadcasts) in { b =>
        b.broadcast(_ + _)
      }
    }
  }

  performance of "low-level same-shape multiplication" in {
    val arrays = numElements.map(n => ArrayNd.fill[Int](n)(10))
    val broadcasts = arrays.map(arr => BroadcastedArrayNd[Int](arr, arr))
    measure method "evaluate" in {
      using(broadcasts) in { b =>
        b.broadcast(_ + _)
      }
    }
  }

  //  performance of "broadcasted array-singleton multiplication" in {
  //    val elements = 1000
  //    val dimensions = Gen.range("dimensions")(1, 10, 1)
  //    val shapes = dimensions.map(n => elements +: Seq.fill(n)(1))
  //    val arr = ArrayNd.fill(elements)(1)
  //    val operand = shapes.map(shape => arr.reshape(shape: _*))
  //    measure method "+" in {
  //      using(operand) in { op =>
  //        op + 1
  //      }
  //    }
  //  }
  //
  //  performance of "broadcasted array-array multiplication" in {
  //    val elements = 1000
  //    val dimensions = Gen.range("dimensions")(0, 9, 1)
  //    val shapes = dimensions.map(n => (elements +: Seq.fill(n)(1), Seq.fill(n)(1) :+ elements))
  //    val arr = ArrayNd.fill(elements)(1)
  //    val operands = shapes.map(pair => (arr.reshape(pair._1: _*), arr.reshape(pair._2: _*)))
  //    measure method "+" in {
  //      using(operands) in { ops =>
  //        ops._1 + ops._2
  //      }
  //    }
  //  }

}
