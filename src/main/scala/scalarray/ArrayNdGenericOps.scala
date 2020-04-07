package scalarray

import scala.reflect.ClassTag

trait ArrayNdGenericOps[@specialized(Int, Double) A] { self: ArrayNdOps[A] =>

  def +(that: ArrayNd[A])(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcast(that)(implicitly[Numeric[A]].plus)
  }
}
