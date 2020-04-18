package scalarray

import scala.reflect.ClassTag

trait ArrayNdNumericOps[@specialized(Int, Long, Float, Double) A] { self: ArrayNdOps[A] =>

  /** Element-wise addition */
  def +(that: ArrayNd[A])(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcastWith(that)(implicitly[Numeric[A]].plus)
  }

  /** Element-wise subtraction */
  def -(that: ArrayNd[A])(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcastWith(that)(implicitly[Numeric[A]].minus)
  }

  /** Element-wise multiplication */
  def *(that: ArrayNd[A])(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcastWith(that)(implicitly[Numeric[A]].times)
  }

  /** Element-wise division */
  def /(that: ArrayNd[A])(implicit integral: Integral[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcastWith(that)(implicitly[Integral[A]].quot)
  }

  /** Element-wise addition */
  def %(that: ArrayNd[A])(implicit integral: Integral[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.broadcastWith(that)(implicitly[Integral[A]].rem)
  }

  /** Element-wise negation */
  def unary_-(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.map(implicitly[Numeric[A]].negate)
  }

  /** Element-wise absolute value */
  def abs(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
    self.map(implicitly[Numeric[A]].abs)
  }
}
