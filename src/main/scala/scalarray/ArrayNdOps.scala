package scalarray

import scala.annotation.tailrec
import scala.collection.AbstractIterator
import scala.reflect.ClassTag

/**
  * N-dimensional array implementation class
  * Supports operations found in collections that return another `ArrayNd` or a single element, which means:
  *   - Operations like `map` return another `ArrayNd` of the same size
  *   - Not __all__ collection operations are supported
  * @tparam A numeric type of element
  */
// TODO should follow signature of other Ops classes when subclasses of ArrayNd are created
trait ArrayNdOps[A] {

  val elements: Array[A]
  val shape: Seq[Int]
  protected val transposed: Boolean

  /** Returns if the array is empty */
  @`inline` def isEmpty: Boolean = elements.isEmpty

  /** Returns if the array is not empty */
  @`inline` def nonEmpty: Boolean = elements.nonEmpty

  /** Returns first element in the array if possible, otherwise throws an exception */
  def head: A = try {
    elements.apply(0)
  } catch {
    case _: ArrayIndexOutOfBoundsException => throw new NoSuchElementException("head of empty array")
  }

  /**
    * Returns last element in the array if possible, otherwise throws an exception
    * Note that the last element in the array is the same regardless of `shape` or `transposed`
    */
  def last: A = {
    try {
      elements(elements.length - 1)
    } catch {
      case _: ArrayIndexOutOfBoundsException => throw new NoSuchElementException("last of empty array")
    }
  }

  /** Returns first element in the array as option, otherwise returns None */
  def headOption: Option[A] = if (isEmpty) None else Some(head)

  /** Returns last element in the array as option, otherwise returns None */
  def lastOption: Option[A] = if (isEmpty) None else Some(last)

  /** Iterator that returns elements in row-major order regardless of whether the array is transposed */
  def iterator: Iterator[A] =  new AbstractIterator[A] {
    private var counter = 0
    private var idx = 0
    private val maxCount = elements.length
    private lazy val indices = Array.fill[Int](shape.length.max(1))(idx)

    private lazy val products = {
      val reverse = shape.reverse
      (shape.length until 0 by -1).map(n => reverse.drop(n).product)
    }

    private def updateState(): Unit = {
      @tailrec
      def update1dIdx(dim: Int): Unit = if (indices(dim) < shape(dim) - 1) {
        indices(dim) += 1
        idx += products.slice(dim, dim + 2).sum
        if (idx >= elements.length) {
          idx = idx - elements.length
        }
      } else {
        indices(dim) = 0
        update1dIdx(dim - 1)
      }

      counter += 1
      if (transposed && hasNext) {
        update1dIdx(indices.length - 1)
      } else {
        idx += 1
      }
    }

    override def hasNext: Boolean = counter < maxCount

    override def next(): A = {
      val element = elements(idx)
      updateState()
      element
    }
  }

  /**
    * Map operation for n-dimensional arrays, which must return another n-dimensional array of the same shape
    * Note that if the array was transposed, the transposed flag will be reset
    *
    * @param f function to apply to each element
    * @tparam B numeric type of element
    * @return array with elements of type `B`
    */
  def map[B: Numeric: ClassTag](f: A => B): ArrayNd[B] = {
    val len = elements.length
    val dest = new Array[B](len)

    iterator.zipWithIndex.foreach {
      case (a, idx) => dest(idx) = f(a)
    }
    val newShape = if (transposed) shape.reverse else shape
    ArrayNd.fromArray[B](dest).reshape(newShape: _*)
  }

}
