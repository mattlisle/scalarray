package scalarray

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * N-Dimensional array implemented by taking views of a 1-dimensional array
  * Appending an entry to `shape` can be thought of as adding another dimension
  * Indexing operations are performed from head to tail on the shape, making a 2D `NdArray` row-major
  * For efficiency, changes to an `NdArray` are done in place, so immutability is not maintained
  *
  * @param elements `NdArray` content in raw, 1D form
  * @param shape sequence of dimension sizes, the product of which must equal the number of elements
  * @param transposed if the matrix is row-major (default) or column major (when transposed)
  * @tparam T any numeric type
  */
class ArrayNd[T: Numeric] private (
  val elements: Array[T],
  val shape: Seq[Int],
  private val transposed: Boolean
) {

  private val length: Int = elements.length

  require(shape.product == length, s"Invalid shape for $length elements: $shape")

  // TODO make an elements class that extends IndexedSeq
  /** Iterates over elements in row-major order */
  protected def elementsIterator: Iterator[T] = new Iterator[T] {
    private var counter = 0
    private var idx = 0
    private val maxCount = elements.length
    private val indices = Array.fill[Int](shape.length.max(1))(idx)

    private val products = {
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

    override def next(): T = {
      val element = elements(idx)
      updateState()
      element
    }
  }

  /**
    * Returns a new array with the elements copied
    *
    * @param newShape for the new array
    */
  protected def withNewShape(newShape: Seq[Int])(implicit classTag: ClassTag[T]): ArrayNd[T] = if (!transposed) {
    new ArrayNd(elements, newShape, transposed)
  } else {
    val copiedElements = {
      val buf = new ArrayBuffer[T]()
      elementsIterator.foreach(elem => buf += elem)
      buf.toArray
    }
    new ArrayNd(copiedElements, newShape, transposed = false)
  }

  /** 1D representation of the array */
  def flatten(implicit classTag: ClassTag[T]): ArrayNd[T] = this.withNewShape(Seq(length))

  /**
    * Takes a view on same underlying data
    *
    * @param dimensions new shape of array which may include up to 1 free dimension
    */
  def reshape(dimensions: Int*)(implicit classTag: ClassTag[T]): ArrayNd[T] = {
    require(dimensions.count(size => size == -1 || size == 0) <= 1, s"Only one free dimension allowed")

    val freeIndex = dimensions.indexOf(-1)
    val partialLength = dimensions.filterNot(_ == -1).product
    val errorMessage = s"Cannot fit $length elements into shape $dimensions"

    val newShape = if (freeIndex >= 0) {
      require(partialLength <= length && length % partialLength == 0 || length == 0, errorMessage)
      dimensions.updated(freeIndex, length / partialLength)
    } else {
      require(partialLength == length, errorMessage)
      dimensions
    }

    this.withNewShape(newShape)
  }

  /** Transpose matrix according to same definition as used in numpy */
  def transpose: ArrayNd[T] = if (shape.length == 1) {
    this
  } else {
    val newShape = shape.reverse
    new ArrayNd(elements, newShape, !transposed)
  }

  override def equals(that: Any): Boolean = that match {
    case array: ArrayNd[_] =>
      this.shape == array.shape &&
      this.elementsIterator.zip(array.elementsIterator).forall {
        case (thisElement, thatElement) => thisElement == thatElement
      }
    case _ => false
  }

  override def toString: String = {
    var nSpaces = 0
    val stringBuilder = new StringBuilder
    val intervalBuf = new ArrayBuffer[Int]()

    def getNumDigits(x: T): Int = if (x == 0) {
      1
    } else {
      (math.log10(implicitly[Numeric[T]].toDouble(x)) + 1).floor.toInt.abs
    }
    val maxDigits = getNumDigits(elements.max)

    @tailrec
    def getIntervals(shapeSlice: Seq[Int]): Seq[Int] = shapeSlice match {
      case Seq(dim) =>
        intervalBuf += dim
        intervalBuf.toSeq
      case Seq(_, dims @ _*) =>
        intervalBuf += shapeSlice.product
        getIntervals(dims)
    }
    val intervals = getIntervals(shape).reverse.to(LazyList)

    def addNewline(): Unit = {
      stringBuilder += '\n'
      Range(0, nSpaces).foreach(_ => stringBuilder += ' ')
    }
    def addOpenParen(): Unit = {
      stringBuilder += '('
      nSpaces += 2
      addNewline()
    }
    def addCloseParen(): Unit = {
      nSpaces -= 2
      addNewline()
      stringBuilder += ')'
    }
    def addOpenBracket(): Unit = {
      stringBuilder += '['
      nSpaces += 1
    }
    def addCloseBracket(): Unit = {
      stringBuilder += ']'
      nSpaces -= 1
    }

    def addElement(element: T, idx: Int): Unit = {
      val remainders = intervals.map((idx + 1) % _).zipWithIndex
      Range(getNumDigits(element), maxDigits).foreach(_ => stringBuilder += ' ')
      stringBuilder ++= element.toString

      /*_*/
      @tailrec
      def addDelimitingChars(lazyList: LazyList[(Int, Int)]): Unit = lazyList match {
        case LazyList() =>
        case (remainder, _) #:: tail if remainder == 0 =>
          addCloseBracket()
          addDelimitingChars(tail)
        case (_, idx) #:: _ =>
          stringBuilder += ','
          if (idx == 0) stringBuilder += ' ' else addNewline()
          if (idx > 1) addNewline()
          Range(0, idx).foreach(_ => addOpenBracket())
      }
      /*_*/
      addDelimitingChars(remainders)
    }

    if (intervals.contains(0)) {
      s"${getClass.getName}([], ${shape.toString})"
    } else {
      stringBuilder ++= getClass.getName
      addOpenParen()
      shape.foreach(_ => addOpenBracket())
      elementsIterator.zipWithIndex.foreach {
        case (element, idx) => addElement(element, idx)
      }
      addCloseParen()
      stringBuilder.toString
    }
  }

}

object ArrayNd {

  /**
    * Factory for homogeneous `ArrayNd` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam T any numeric type
    */
  def fill[T: Numeric: ClassTag](shape: Int*)(elem: => T): ArrayNd[T] = new ArrayNd(
    elements = Array.fill(shape.product)(elem),
    shape = shape,
    transposed = false
  )

  /**
    * Factory for `ArrayNd` from an existing array
    *
    * @param data from source array
    * @tparam T any numeric type
    */
  def fromArray[T: Numeric](data: Array[T]) = new ArrayNd[T](
    elements = data,
    shape = Seq(data.length),
    transposed = false
  )
  
}
