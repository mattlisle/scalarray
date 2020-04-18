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
  * @param _strides override strides definition for non-contiguous arrays
  * @param contiguous equivalent to c-contiguous in numpy, or row-major if the array is 2D
  * @tparam A numeric type
  */
class ArrayNd[@specialized(Int, Long, Float, Double) A: Numeric] (
  val elements: Array[A],
  val shape: Seq[Int],
  override protected val _strides: Option[Seq[Int]],
  override protected val contiguous: Boolean
) extends ArrayNdOps[A] {

  _strides.foreach(s => require(shape.length == s.length, s"Invalid strides $s for shape $shape"))

  require(
    if (_strides.isDefined) {
      shape.zip(_strides.get).map {
        case (dim, stride) => if (stride == 0) 1 else dim
      }.product == elements.length
    } else {
      shape.product == elements.length
    },
    s"Invalid shape for ${elements.length} elements: $shape"
  )

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
    val idx1d = get1dIndex(indices)
    elements(idx1d)
  }

  private def get1dIndex(indexNd: Seq[Int]): Int = {
    @tailrec
    def helper(idxs: Seq[Int], shp: Seq[Int], str: Seq[Int], prev: Int = 0): Int = (idxs, shp, str) match {
      case (Seq(), Seq(), _) =>
        prev
      case (Seq(idx, idxs @ _*), Seq(dim, dims @ _*), Seq(step, steps @ _*)) =>
        val positiveIdx = if (idx >= 0) idx else idx + dim
        helper(idxs, dims, steps, prev + positiveIdx * step)
    }
    helper(indexNd, shape, strides)
  }

  /** 1D representation of the array */
  def flatten(implicit classTag: ClassTag[A]): ArrayNd[A] = this.reshape(elements.length)

  /**
    * Takes a view on same underlying data
    *
    * @param dimensions new shape of array which may include up to 1 free dimension
    */
  def reshape(dimensions: Int*)(implicit classTag: ClassTag[A]): ArrayNd[A] = {
    require(dimensions.count(size => size == -1 || size == 0) <= 1, s"Only one free dimension allowed")

    val len = elements.length
    val freeIndex = dimensions.indexOf(-1)
    val partialLength = dimensions.filterNot(_ == -1).product
    val errorMessage = s"Cannot fit $elements.length elements into shape $dimensions"

    val newShape = if (freeIndex >= 0) {
      require(partialLength <= len && len % partialLength == 0 || len == 0, errorMessage)
      dimensions.updated(freeIndex, len / partialLength)
    } else {
      require(partialLength == len, errorMessage)
      dimensions
    }
    if (contiguous) {
      new ArrayNd(elements, newShape, None, contiguous)
    } else {
      val copiedElements = new Array[A](len)
      var idx = 0
      iterator.foreach { elem =>
        copiedElements(idx) = elem
        idx += 1
      }
      new ArrayNd(copiedElements, newShape, None, contiguous = true)
    }
  }

  /** Transpose matrix according to same definition as used in numpy */
  def transpose: ArrayNd[A] = if (shape.length == 1) {
    this
  } else {
    val newShape = shape.reverse
    val newStrides = _strides.map(_.reverse)
    new ArrayNd(elements, newShape, newStrides, !contiguous)
  }

  override def map[@specialized(Int, Long, Float, Double) B: Numeric: ClassTag](f: A => B): ArrayNd[B] = {
    val len = elements.length
    val dest = new Array[B](len)

    var idx = 0
    iterator.foreach { elem =>
      dest(idx) = f(elem)
      idx += 1
    }
    new ArrayNd(dest, shape, None, contiguous = true)
  }

  override def broadcastTo(thatShape: Seq[Int]): ArrayNd[A] = {
    if (shape == thatShape) {
      this
    } else {
      val (broadcastShape, broadcastStrides) = getBroadcastParams(thatShape)
      new ArrayNd[A](
        elements = elements,
        shape = broadcastShape,
        _strides = Some(broadcastStrides),
        contiguous = false
      )
    }
  }

  override def broadcastWith(that: ArrayNd[A])(f: (A, A) => A)(implicit tag: ClassTag[A]): ArrayNd[A] = {
    val thisBroadcasted = broadcastTo(that.shape)

    val thisIt = broadcastTo(that.shape).iterator
    val thatIt = that.broadcastTo(shape).iterator

    val newShape = thisBroadcasted.shape
    val len = newShape.product
    val newElems: Array[A] = new Array[A](len)

    var idx = 0
    while (idx < len) {
      newElems(idx) = f(thisIt.next(), thatIt.next())
      idx += 1
    }
    new ArrayNd(newElems, newShape, None, contiguous = true)
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
      val asDouble = implicitly[Numeric[A]].toDouble(x)
      val isNegative = asDouble < 0
      (math.log10(asDouble.abs) + 1).floor.toInt + (if (isNegative) 1 else 0)
    }

    val maxDigits = if (elements.nonEmpty) elements.map(getNumDigits).max else 0

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
      s"ArrayNd([], ${shape.toString})"
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

object ArrayNd {

  /**
    * Factory for homogeneous `ArrayNd` of specified shape
    *
    * @param shape sequence of dimension sizes, the product of will be the number of elements
    * @param elem value with which to fill the array
    * @tparam A any numeric type
    */
  def fill[@specialized(Int, Long, Float, Double) A: Numeric: ClassTag](shape: Int*)(elem: => A): ArrayNd[A] = {
    new ArrayNd(
      elements = Array.fill(shape.product)(elem),
      shape = shape,
      _strides = None,
      contiguous = true
    )
  }

  /**
    * Factory for `ArrayNd` from an existing array
    *
    * @param data from source array
    * @tparam A any numeric type
    */
  def fromArray[@specialized(Int, Long, Float, Double) A: Numeric](data: Array[A]) = new ArrayNd[A](
    elements = data,
    shape = Seq(data.length),
    _strides = None,
    contiguous = true
  )

  /**
    * Converts any number to a singleton array
    *
    * @param element will become the singleton
    * @tparam A numeric type
    */
  implicit def fromPrimitive[@specialized(Int, Long, Float, Double) A: Numeric: ClassTag](
    element: A
  ): ArrayNd[A] = new ArrayNd[A](
    elements = Array(element),
    shape = Seq(1),
    _strides = None,
    contiguous = true
  )
  
}
