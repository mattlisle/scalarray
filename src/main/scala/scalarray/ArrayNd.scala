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
  * @param rowMajor if the matrix is row-major (default) or column major (when transposed)
  * @tparam T any numeric type
  */
class ArrayNd[T: Numeric] private (
  val elements: Array[T],
  val shape: Seq[Int],
  private val rowMajor: Boolean
) {

  private val length: Int = elements.length

  private val stride: Int = if (rowMajor) 1 else shape.head

  require(shape.product == length, s"Invalid shape for $length elements: $shape")

  /** Iterates over elements in row-major order */
  protected def elementsIterator: Iterator[T] = new Iterator[T] {
    private var counter = 0
    private var idx = 0

    override def hasNext: Boolean = counter < elements.length

    override def next(): T = {
      if (idx >= elements.length) idx = idx - elements.length + 1
      val element = elements(idx)
      counter += 1
      idx += stride
      element
    }
  }

  /** 1D representation of the array */
  def flatten: ArrayNd[T] = new ArrayNd(elements, Seq(length), rowMajor)

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

    new ArrayNd(elements, newShape, rowMajor)
  }

  /** Transpose matrix according to same definition as used in numpy */
  def transpose: ArrayNd[T] = {
    val newShape = shape.reverse
    new ArrayNd(elements, newShape, !rowMajor)
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
    * Factory for homogeneous `NdArray` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam T any numeric type
    */
  def fill[T: Numeric: ClassTag](shape: Int*)(elem: => T): ArrayNd[T] = new ArrayNd(
    elements = Array.fill(shape.product)(elem),
    shape = shape,
    rowMajor = true
  )
  
}
