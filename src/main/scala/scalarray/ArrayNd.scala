package scalarray

import scala.annotation.tailrec
import scala.collection.ArrayOps
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
  * @tparam A any numeric type
  */
class ArrayNd[A: Numeric] (
  val elements: Array[A],
  val shape: Seq[Int],
  override protected val transposed: Boolean
) extends ArrayNdOps[A] {

  private val length: Int = elements.length

  require(shape.product == length, s"Invalid shape for $length elements: $shape")

  /**
    * Finds an element at the specified indices
    * The indices are applied in the same order as the dimensions listed in `shape`, e.g.
    * - If the array is 2-dimensional, this function would be called as array(row, col)
    * - If the array is 3-dimensional, as in an array of 2D matrices, it would be array(matrix, row, col)
    *
    * @param indices of the desired element in the array
    * @return the element at the specified location
    */
  def apply(indices: Int*): A = {
    require(indices.length == shape.length, s"All dimensions must be specified but was given: $indices")

    @tailrec
    def get1dIndex(idxSlice: Seq[Int], shapeSlice: Seq[Int], prev: Int = 0): Int = (idxSlice, shapeSlice) match {
      case (Seq(), Seq())   => prev
      case (Seq(idx, idxs @ _*), Seq(dim, dims @ _*)) =>
        val positiveIdx = if (idx >= 0) idx else idx + dim
        get1dIndex(idxs, dims, prev + positiveIdx * dims.product)
    }

    val idx1d = if (transposed) get1dIndex(indices.reverse, shape.reverse) else get1dIndex(indices, shape)
    elements(idx1d)
  }

  /**
    * Returns a new array with the elements copied
    *
    * @param newShape for the new array
    */
  protected def withNewShape(newShape: Seq[Int])(implicit classTag: ClassTag[A]): ArrayNd[A] = if (!transposed) {
    new ArrayNd(elements, newShape, transposed)
  } else {
    val copiedElements = {
      val buf = new ArrayBuffer[A]()
      iterator.foreach(elem => buf += elem)
      buf.toArray
    }
    new ArrayNd(copiedElements, newShape, transposed = false)
  }

  /** 1D representation of the array */
  def flatten(implicit classTag: ClassTag[A]): ArrayNd[A] = this.withNewShape(Seq(length))

  /**
    * Takes a view on same underlying data
    *
    * @param dimensions new shape of array which may include up to 1 free dimension
    */
  def reshape(dimensions: Int*)(implicit classTag: ClassTag[A]): ArrayNd[A] = {
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
  def transpose: ArrayNd[A] = if (shape.length == 1) {
    this
  } else {
    val newShape = shape.reverse
    new ArrayNd(elements, newShape, !transposed)
  }

  override def equals(that: Any): Boolean = that match {
    case array: ArrayNd[_] =>
      this.shape == array.shape &&
      this.iterator.zip(array.iterator).forall {
        case (thisElement, thatElement) => thisElement == thatElement
      }
    case _ => false
  }

  override def toString: String = {
    var nSpaces = 0
    val stringBuilder = new StringBuilder
    val intervalBuf = new ArrayBuffer[Int]()

    def getNumDigits(x: A): Int = if (x == 0) {
      1
    } else {
      (math.log10(implicitly[Numeric[A]].toDouble(x)) + 1).floor.toInt.abs
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

    def addElement(element: A, idx: Int): Unit = {
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
      iterator.zipWithIndex.foreach {
        case (element, idx) => addElement(element, idx)
      }
      addCloseParen()
      stringBuilder.toString
    }
  }
}

object ArrayNd extends ArrayNdFactory {

  /**
    * Factory for homogeneous `ArrayNd` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam A any numeric type
    */
  override def fill[A: Numeric: ClassTag](shape: Int*)(elem: => A): ArrayNd[A] = new ArrayNd(
    elements = Array.fill(shape.product)(elem),
    shape = shape,
    transposed = false
  )

  /**
    * Factory for `ArrayNd` from an existing array
    *
    * @param data from source array
    * @tparam A any numeric type
    */
  override def fromArray[A: Numeric](data: Array[A]): ArrayNd[A] = new ArrayNd[A](
    elements = data,
    shape = Seq(data.length),
    transposed = false
  )

  implicit def fromPrimitive[A: Numeric: ClassTag](element: A): ArrayNd[A] = new ArrayNd[A](
    elements = Array(element),
    shape = Seq(1),
    transposed = false
  )
  
}
