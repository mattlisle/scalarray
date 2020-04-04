package scalarray

import scala.reflect.ClassTag

trait ArrayNdFactory {

  /**
    * Factory for homogeneous `ArrayNd` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam A any numeric type
    */
  def fill[A: Numeric: ClassTag](shape: Int*)(elem: => A): ArrayNd[A]

  /**
    * Factory for `ArrayNd` from an existing array
    *
    * @param data from source array
    * @tparam A any numeric type
    */
  def fromArray[A: Numeric](data: Array[A]): ArrayNd[A]

  /**
    * Convert any numeric element into an array
    *
    * @param element to convert into a singleton array
    * @tparam A any numeric type
    */
  def fromPrimitive[A: Numeric: ClassTag](element: A): ArrayNd[A]

}
