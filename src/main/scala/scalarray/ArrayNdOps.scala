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

  /** Iterator returning pairs of elements from this array broadcast onto another */
  def broadcast(that: ArrayNd[A])(f: (A, A) => A)(implicit numeric: Numeric[A], classTag: ClassTag[A]): ArrayNd[A] = {
//    val it = that match {
//      case MatchingBroadcastIterator(_) => new MatchingBroadcastIterator(that)
//      case StandardBroadcastIterator(_) => new StandardBroadcastIterator(that)
//      case _ => throw new Exception
//    }
    var idx = 0
    val len = size
    val elems = new Array[A](len)
    val those = that.elements
    while (idx < len) {
      elems(idx) = f(elements(idx), those(idx))
      idx += 1
    }
//    val elems = new Array[A](it.broadcastShape.product)
//    it.foreach { pair =>
//      elems(idx) = f(pair._1, pair._2)
//      idx += 1
//    }
    new ArrayNd[A](elems, this.shape, transposed = false)
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
    private val shapeArray = shape.toArray

    private val products: Array[Int] = {
      val reverse = shape.reverse
      (shape.length until 0 by -1).map(n => reverse.drop(n).product)
      }.toArray


    private def updateState(): Unit = {
      @tailrec
      def update1dIdx(dim: Int): Unit = if (indices(dim) < shapeArray(dim) - 1) {
        indices(dim) += 1
        idx += products(dim) + products.lift(dim + 1).getOrElse(0)
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

  abstract class BroadcastIterator(val that: ArrayNd[A]) extends AbstractIterator[(A, A)] {
    val broadcastShape: Seq[Int]
  }

  sealed trait BroadcastIteratorExtractor {
    def validateDims(thisShape: Seq[Int], thatShape: Seq[Int]): Boolean
    def unapply(that: ArrayNd[A]): Option[ArrayNd[A]] =  if (validateDims(shape, that.shape)) {
      Some(that)
    } else {
      None
    }
  }

  class MatchingBroadcastIterator(that: ArrayNd[A]) extends BroadcastIterator(that) {
    require(shape == that.shape)
    override val broadcastShape: Seq[Int] = shape
    private var counter = 0
    private var idx = 0
    private val maxCount = elements.length

    override def hasNext: Boolean = counter < maxCount

    override def next(): (A, A) = {
      val element = (elements(idx), that.elements(idx))
      counter += 1
      idx += 1
      element
    }
  }
  case object MatchingBroadcastIterator extends BroadcastIteratorExtractor {
    override def validateDims(thisShape: Seq[Int], thatShape: Seq[Int]): Boolean = thisShape == thatShape
  }

  /** Iterator for non-transposed `ArrayNd` that's broadcast from `oldShape` to `newShape` */
  class StandardBroadcastIterator(that: ArrayNd[A]) extends BroadcastIterator(that) {

    val (broadcastShape, thisShape, thatShape, thisIntervals, thatIntervals) = getShapes

    private var broadcast1dIdx = 0
    private var this1dIdx = 0
    private var that1dIdx = 0

    private val thisIndices = Array.fill[Int](broadcastShape.length)(0)
    private val thatIndices = Array.fill[Int](broadcastShape.length)(0)

    private val maxCount = broadcastShape.product
    private val maxDim = broadcastShape.length - 1

    def getShapes: (Seq[Int], Array[Int], Array[Int], Array[Int], Array[Int]) = {
      val thisLength = shape.length
      val thatLength = that.shape.length

      val (thisPadded, thatPadded) = if (thisLength > thatLength) {
        (shape, that.shape.padTo(thisLength, 1))
      } else {
        (shape.padTo(thatLength, 1), that.shape)
      }

      val broadcastShapeBuf = new ListBuffer[Int]()

      @tailrec
      def compareDims(x: Seq[Int], y: Seq[Int]): Unit = (x, y) match {
        case (Seq(x, xs @ _*), Seq(y, ys @ _*)) =>
          if (x == y || x == 1 || y == 1) {
            broadcastShapeBuf += x.max(y)
            compareDims(xs, ys)
          } else {
            throw new Exception // BroadcastException(shape, that.shape)
          }
        case (Seq(), Seq()) =>
      }

      compareDims(thisPadded, thatPadded)
      val (thisIntervals, thatIntervals) = (getDimIntervals(thisPadded), getDimIntervals(thatPadded))
      (broadcastShapeBuf.toSeq, thisPadded.toArray, thatPadded.toArray, thisIntervals, thatIntervals)
    }

    def updateState(): Unit = {

      @tailrec
      def updateThis(dim: Int, thisDim: Int, thatDim: Int): Unit = {
        if (thisDim == 1 && thatDim > 1 && thatIndices(dim) < thatDim - 1) {
          this1dIdx -= thisIntervals(dim) - 1
        } else if (thisIndices(dim) < thisDim - 1) {
          thisIndices(dim) += 1
          this1dIdx += 1
        } else {
          thisIndices(dim) = 0
          val nextDim = dim - 1
          updateThis(nextDim, thisShape(nextDim), thatShape(nextDim))
        }
      }

      @tailrec
      def updateThat(dim: Int, thisDim: Int, thatDim: Int): Unit = {
        if (thatDim == 1 && thisDim > 1 && thisIndices(dim) > 0) {
          that1dIdx -= thatIntervals(dim) - 1
        } else if (thatIndices(dim) < thatDim - 1) {
          thatIndices(dim) += 1
          that1dIdx += 1
        } else {
          thatIndices(dim) = 0
          val nextDim = dim - 1
          updateThat(nextDim, thisShape(nextDim), thatShape(nextDim))
        }
      }

      broadcast1dIdx += 1
      if (hasNext) {
        updateThis(maxDim, thisShape(maxDim), thatShape(maxDim))
        updateThat(maxDim, thisShape(maxDim), thatShape(maxDim))
      }
    }

    override def hasNext: Boolean = broadcast1dIdx < maxCount

    override def next(): (A, A) = {
      val pair = (elements(this1dIdx), that.elements(that1dIdx))
      updateState()
      pair
    }
  }
  case object StandardBroadcastIterator extends BroadcastIteratorExtractor {
    override def validateDims(thisShape: Seq[Int], thatShape: Seq[Int]): Boolean = {
      val thisLength = thisShape.length
      val thatLength = thatShape.length

      val (thisPadded, thatPadded) = if (thisLength > thatLength) {
        (thisShape, thatShape.padTo(thisLength, 1))
      } else {
        (shape.padTo(thatLength, 1), thatShape)
      }

      @tailrec
      def compareDims(x: Seq[Int], y: Seq[Int]): Boolean = (x, y) match {
        case (Seq(x, xs @ _*), Seq(y, ys @ _*)) =>
          if (x == y || x == 1 || y == 1) {
            compareDims(xs, ys)
          } else {
            false
          }
        case (Seq(), Seq()) => true
        case _ => false
      }
      compareDims(thisPadded, thatPadded)
    }
  }

  protected def getDimIntervals(someShape: Seq[Int]): Array[Int] = {
    val intervalBuf = new ArrayBuffer[Int]
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
    intervalBuf.toArray
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
