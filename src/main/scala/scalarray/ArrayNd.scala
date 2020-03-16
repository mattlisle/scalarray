package scalarray

import scala.reflect.ClassTag

/**
  * N-Dimensional array implemented by taking views of a 1-dimensional array
  * Appending an entry to `shape` can be thought of as adding another dimension
  * Indexing operations are performed from head to tail on the shape, making a 2D `NdArray` row-major
  * For efficiency, changes to an `NdArray` are done in place, so immutability is not maintained
  *
  * @param elements `NdArray` content in raw, 1D form
  * @param shape sequence of dimension sizes, the product of which must equal the number of elements
  * @tparam T any numeric type
  */
class ArrayNd[T: Numeric] private (
  private val elements: Array[T],
  val shape: Seq[Int] 
) {

  private val length: Int = elements.length

  require(shape.product == length, s"Invalid shape for $length elements: $shape")

  /** 1D representation of the array */
  def flatten: ArrayNd[T] = new ArrayNd(elements, Seq(length))

  /**
    * Takes a view on same underlying data
    *
    * @param partialShape new shape of array which may include up to 1 free dimension
    */
  def reshape(partialShape: Int*): ArrayNd[T] = {
    require(partialShape.count(size => size == -1 || size == 0) <= 1, s"Only one free dimension allowed")

    val freeIndex = partialShape.indexOf(-1)
    val partialLength = partialShape.filterNot(_ == -1).product
    val errorMessage = s"Cannot fit $length elements into shape $partialShape"

    val newShape = if (freeIndex >= 0) {
      require(partialLength <= length && length % partialLength == 0 || length == 0, errorMessage)
      partialShape.updated(freeIndex, length / partialLength)
    } else {
      require(partialLength == length, errorMessage)
      partialShape
    }

    new ArrayNd(elements, newShape)
  }

}

object ArrayNd {

  /**
    * Factory for homogeneous `NdArray` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam T any numeric type
    */
  def fill[T: Numeric: ClassTag](shape: Int*)(elem: => T): ArrayNd[T] = new ArrayNd(
    elements = Array.fill(shape.product)(elem),
    shape = shape
  )

}
