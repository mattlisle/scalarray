package scalarray

import scala.reflect.ClassTag

trait ArrayNdGenericOps[A] { self: ArrayNdOps[A] =>

  def +(that: ArrayNd[A])(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    val elements = new Array[A](self.size)
    var idx = 0
    self.broadcast(that).foreach { pair =>
      elements(idx) = implicitly[Numeric[A]].plus(pair._1, pair._2)
      idx += 1
    }
    new ArrayNd[A](elements, this.shape, transposed = false)
  }
}
