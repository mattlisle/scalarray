package scalarray

import scala.annotation.tailrec
import scala.collection.AbstractIterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.reflect.ClassTag

/**
  * N-dimensional array implementation class
  * Supports operations found in collections that return another `ArrayNd` or a single element, which means:
  *   - Operations like `map` return another `ArrayNd` of the same size
  *   - Not __all__ collection operations are supported
  * @tparam A numeric type of element
  */
// TODO should follow signature of other Ops classes when subclasses of ArrayNd are created
trait ArrayNdOps[@specialized(Char, Int, Long, Float, Double) A] {

  val elements: Array[A]
  val shape: Seq[Int]
  protected val _strides: Option[Seq[Int]]
  protected val contiguous: Boolean

  /**
    * Defined such that the inner product of this and a given ND index yield the 1D index in `elements`
    * @return strides corresponding to the shape of this array
    */
  def strides: Seq[Int] = _strides.getOrElse {
    if (contiguous) {
      shape.indices.map(n => shape.drop(n + 1).product)
    } else {
      shape.indices.map(n => shape.take(n).product)
    }
  }

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
  def iterator: Iterator[A] = if (contiguous) new ContiguousIterator else new NonContiguousIterator

  /**
    * Map operation for n-dimensional arrays, which must return another n-dimensional array of the same shape
    * Note that if the array was transposed, the transposed flag will be reset
    *
    * @param f function to apply to each element
    * @tparam B numeric type of element
    * @return array with elements of type `B`
    */
  def map[@specialized(Char, Int, Long, Float, Double) B: Numeric: ClassTag](f: A => B): ArrayNd[B]

  /**
    * Return a new array with the same underlying elements and a new shape
    * @param thatShape to broadcast to
    */
  def broadcastTo(thatShape: Seq[Int]): ArrayNd[A]

  /**
    * Operates on broadcasted array with a specified function
    *
    * @param that array to broadcast with
    * @param f operation to perform on pairs of elements
    * @return new n-dimensional array
    */
  def broadcastWith(that: ArrayNd[A])(f: (A, A) => A)(implicit tag: ClassTag[A]): ArrayNd[A]

  /**
    * Gets parameters needed to build an array broadcasted to the provided shape
    * @param thatShape shape that we're broadcasting to
    * @return shape and strides of the broadcasted array
    */
  protected def getBroadcastParams(thatShape: Seq[Int]): (Seq[Int], Seq[Int]) = {
    @tailrec
    def buildResults(
      thisShp: Seq[Int],
      thatShp: Seq[Int],
      thisStrides: Seq[Int],
      broadcastStrides: Seq[Int],
      broadcastShape: Seq[Int]
    ): (Seq[Int], Seq[Int]) = (thisShp, thatShp) match {
      case (Seq(x, xs @ _*), Seq(y, ys @ _*)) =>
        if (x == y || x == 1 || y == 1) {
          val nextStride = if (x == 1 && y != 1) 0 else thisStrides.head
          buildResults(xs, ys, thisStrides.tail, nextStride +: broadcastStrides, x.max(y) +: broadcastShape)
        } else {
          throw new Exception
        }
      case (Seq(x, xs @ _*), empty @ Seq()) =>
        buildResults(xs, empty, thisStrides.tail, thisStrides.head +: broadcastStrides, x +: broadcastShape)
      case (empty @ Seq(), Seq(y, ys @ _*)) =>
        buildResults(empty, ys, empty, 0 +: broadcastStrides ,y +: broadcastShape)
      case _ => (broadcastShape, broadcastStrides)
    }

    buildResults(shape.reverse, thatShape.reverse, strides.reverse, Seq[Int](), Seq[Int]())
  }

  /** Iterator for a standard (non-transposed) `ArrayNd` */
  private class ContiguousIterator extends AbstractIterator[A] {
    protected var idx: Int = -1
    protected lazy val maxCount: Int = elements.length - 1

    override def hasNext: Boolean = idx < maxCount

    override def next(): A = {
      idx += 1
      elements(idx)
    }
  }

  /** Iterator for a transposed `ArrayNd` */
  private class NonContiguousIterator extends AbstractIterator[A] {
    private var counter = 0
    private var idx = 0
    private val maxCount: Int = shape.product
    private val indices = Array.fill[Int](shape.length.max(1))(idx)

    private val shapeArray = new Array[Int](shape.length)
    shape.copyToArray(shapeArray)
    private val stridesArray: Array[Int] = new Array[Int](shape.length)
    strides.copyToArray(stridesArray)

    private def updateState(): Unit = {
      @tailrec
      def update1dIdx(dim: Int): Unit = if (indices(dim) < shapeArray(dim) - 1) {
        indices(dim) += 1
        idx += stridesArray(dim)
      } else {
        idx -= stridesArray(dim) * indices(dim)
        indices(dim) = 0
        update1dIdx(dim - 1)
      }

      counter += 1
      if (hasNext) update1dIdx(indices.length - 1)
    }

    override def hasNext: Boolean = counter < maxCount

    override def next(): A = {
      val element = elements(idx)
      updateState()
      element
    }
  }
}
