package scalarray

import scala.annotation.tailrec
import scala.collection.AbstractIterator
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.reflect.ClassTag
import scalarray.ArrayNdOps.BroadcastException

/**
  * N-dimensional array implementation class
  * Supports operations found in collections that return another `ArrayNd` or a single element, which means:
  *   - Operations like `map` return another `ArrayNd` of the same size
  *   - Not __all__ collection operations are supported
  * @tparam A numeric type of element
  */
// TODO should follow signature of other Ops classes when subclasses of ArrayNd are created
trait ArrayNdOps[A] extends ArrayNdGenericOps[A] {

  val elements: Array[A]
  val shape: Seq[Int]
  protected val transposed: Boolean

  /** The size of this array */
  @`inline` def size: Int = elements.length

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
  def iterator: Iterator[A] = if (transposed) new TransposedIterator else new StandardIterator

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

  /**
    * Foreach on an `ArrayNd` iterates over the iterator
    *
    * @param f function to apply to each element
    */
  def foreach(f: A => Unit): Unit = iterator.foreach(f)


  /** Iterator for a standard (non-transposed) `ArrayNd` */
  private class StandardIterator extends AbstractIterator[A] {
    private var counter = 0
    private var idx = 0
    private val maxCount = elements.length

    override def hasNext: Boolean = counter < maxCount

    override def next(): A = {
      val element = elements(idx)
      counter += 1
      idx += 1
      element
    }
  }

  /** Iterator for a transposed `ArrayNd` */
  private class TransposedIterator extends AbstractIterator[A] {
    private var counter = 0
    private var idx = 0
    private val maxCount = elements.length
    private val indices = Array.fill[Int](shape.length.max(1))(idx)

    private val products: Array[Int] = {
      val reverse = shape.reverse
      (shape.length until 0 by -1).map(n => reverse.drop(n).product)
      }.toArray


    private def updateState(): Unit = {
      @tailrec
      def update1dIdx(dim: Int): Unit = if (indices(dim) < shape(dim) - 1) {
        indices(dim) += 1
        idx += products(dim) + products(dim + 1)
        if (idx >= elements.length) {
          idx = idx - elements.length
        }
      } else {
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

  /** Iterator for non-transposed `ArrayNd` that's broadcast from `oldShape` to `newShape` */
  private class StandardBroadcastIterator(val that: ArrayNd[A]) extends AbstractIterator[A] {

    def getBroadcastShape: Seq[Int] = {
      val thisLength = shape.length
      val thatLength = that.shape.length

      val (thisPadded, thatPadded) = if (thisLength > thatLength) {
        (shape, that.shape.padTo(thisLength, 1))
      } else {
        (shape.padTo(thatLength, 1), that.shape)
      }

      val broadcastShape = new ListBuffer[Int]

      @tailrec
      def compareDims(x: Seq[Int], y: Seq[Int]): Unit = (x, y) match {
        case (x :: xs, y :: ys) =>
          if (x != y && (x != 1 || y != 1)) {
            throw new BroadcastException(shape, that.shape)
          } else {
            broadcastShape += x.max(y)
            compareDims(xs, ys)
          }
        case _ =>
      }

      compareDims(thisPadded, thatPadded)
      broadcastShape.toSeq
    }

    def getBounds(shp: Seq[Int]): Array[Int] = {
      val buf = new ArrayBuffer[Int]
      buf += 0

      buf.toArray
    }

    private val broadcastShape = getBroadcastShape

    private val thisIntervals = getDimIntervals(shape)
    private val thatIntervals = getDimIntervals(that.shape)

    private var thisIdx = 0
    private var thatIdx = 0
    private var broadcastIdx = 0

    private var counter = 0
    private val maxCount = broadcastShape.product
    private val indices = Array.fill[Int](broadcastShape.length)(broadcastIdx)

    override def hasNext: Boolean = ???

    override def next(): A = ???
  }

  protected def getDimIntervals(someShape: Seq[Int]): Seq[Int] = {
    val intervalBuf = new ListBuffer[Int]
    @tailrec
    def getIntervals(shapeSlice: Seq[Int]): Seq[Int] = shapeSlice match {
      case Seq(dim)        =>
        intervalBuf += dim
        intervalBuf.toSeq
      case Seq(_, dims@_*) =>
        intervalBuf += shapeSlice.product
        getIntervals(dims)
    }
    getIntervals(someShape)
    intervalBuf.toSeq
  }

  protected def getCopyIfTransposed(
    implicit
    numeric: Numeric[A],
    classTag: ClassTag[A]
  ): ArrayNd[A] = if (!transposed) {
    new ArrayNd(elements, shape, transposed)
  } else {
    val copiedElements = {
      val buf = new ArrayBuffer[A]()
      iterator.foreach(elem => buf += elem)
      buf.toArray
    }
    new ArrayNd(copiedElements, shape, transposed = false)
  }

}

object ArrayNdOps {

  class BroadcastException(shape1: Seq[Int], shape2: Seq[Int])
    extends IllegalArgumentException(s"Cannot broadcast array of shape $shape2 into $shape1")

}
