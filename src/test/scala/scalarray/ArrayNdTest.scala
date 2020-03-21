package scalarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ArrayNdTest extends AnyFlatSpec with Matchers {

  behavior of "indexing an array"

  it should "throw an exception if not enough indices are supplied" in {
    val arr = ArrayNd.fill[Int](4, 4)(0)
    an[IllegalArgumentException] should be thrownBy arr(0)
  }

  it should "index a negative dimension correctly" in {
    val arr = ArrayNd.fromArray((0 until 5).toArray)
    arr(-2) shouldEqual 3
  }

  it should "index a multidimensional array correctly" in {
    val arr = ArrayNd.fromArray((0 until 24).toArray).reshape(3, 2, 4)
    arr(1, 0, 2) shouldEqual 10
  }

  it should "index a multidimensional array transpose correctly" in {
    val arr = ArrayNd.fromArray((0 until 24).toArray).reshape(3, 2, 4).transpose
    arr(1, 0, 2) shouldEqual 17
  }


  behavior of "reshaping an array"

  it should "fail to reshape an array with multiple free dimensions" in {
    val arr = ArrayNd.fill[Int](20)(0)
    an[IllegalArgumentException] should be thrownBy arr.reshape(1, -1, -1)
  }

  it should "reshape an array to a valid, fully-specified shape" in {
    val arr = ArrayNd.fill[Int](20)(0)
    arr.reshape(5, 4).shape shouldEqual Seq(5, 4)
  }

  it should "reshape an array with one free axis and other axes whose product is a factor of the array length" in {
    val arr = ArrayNd.fill[Int](20)(0)
    arr.reshape(5, 2, -1).shape shouldEqual Seq(5, 2, 2)
  }

  it should "fail to reshape an array with one free axis when the array cannot be equally divided" in {
    val arr = ArrayNd.fill[Int](20)(0)
    an[IllegalArgumentException] should be thrownBy arr.reshape(3, 2, -1)
  }

  it should "reshape an zero-length array to an arbitrary shape with the last dimension having zero size" in {
    val arr = ArrayNd.fill[Int](0)(0)
    arr.reshape(1, 0).shape shouldEqual Seq(1, 0)
  }

  it should "reshape a zero-length array to an arbitrary shape with any dimension having zero size" in {
    val arr = ArrayNd.fill[Int](0)(0)
    arr.reshape(5, 0, 4).shape shouldEqual Seq(5, 0, 4)
  }

  it should "reshape a zero-length array to an arbitrary shape with one free and no zero dimensions" in {
    val arr = ArrayNd.fill[Int](0)(0)
    arr.reshape(5, 4, -1).shape shouldEqual Seq(5, 4, 0)
  }

  it should "fail to reshape an array into a shape with a zero dimension and a free dimension" in {
    val arr = ArrayNd.fill[Int](0)(0)
    an[IllegalArgumentException] should be thrownBy arr.reshape(5, 0, -1)
  }

  it should "fail to reshape an array into a shape with multiple zero dimensions" in {
    val arr = ArrayNd.fill[Int](0)(0)
    an[IllegalArgumentException] should be thrownBy arr.reshape(5, 0, 0)
  }

  behavior of "printing an array"

  it should "correctly print out a 2 by 2 matrix" in {
    val arr = ArrayNd.fill[Int](2, 2)(0)
    arr.toString shouldEqual
      s"""${arr.getClass.getName}(
         |  [[0, 0],
         |   [0, 0]]
         |)""".stripMargin
  }

  it should "correctly print out an array of shape (2,)" in {
    val arr = ArrayNd.fill[Int](2)(0)
    val expected = s"""${arr.getClass.getName}(
                      |  [0, 0]
                      |)""".stripMargin
    val actual = arr.toString
    actual shouldEqual expected
  }

  it should "correctly print out a 2 by 1 matrix" in {
    val arr = ArrayNd.fill[Int](2, 1)(0)
    arr.toString shouldEqual
      s"""${arr.getClass.getName}(
         |  [[0],
         |   [0]]
         |)""".stripMargin
  }

  it should "correctly print out a 2 by 2 by 2 tensor" in {
    val arr = ArrayNd.fill[Int](2, 2, 2)(0)
    arr.toString.stripMargin shouldEqual
      s"${arr.getClass.getName}(\n  [[[0, 0],\n    [0, 0]],\n   \n   [[0, 0],\n    [0, 0]]]\n)"
  }

  behavior of "transposing an array"

  private val transposeArr2d = ArrayNd.fromArray(Array(1, 2, 3, 4)).reshape(2, 2)
  private val transposeArr3d = ArrayNd.fromArray((0 until 24).toArray).reshape(3, 2, 4)

  it should "equal the tranpose of the transpose of itself" in {
    transposeArr2d.transpose.transpose shouldEqual transposeArr2d
  }

  it should "retain the correct order after flattening after a transpose" in {
    transposeArr2d.transpose.flatten shouldEqual ArrayNd.fromArray(Array(1, 3, 2, 4))
  }

  it should "always return the same array for the transpose of a 1D array" in {
    transposeArr2d.transpose.flatten.transpose shouldEqual ArrayNd.fromArray(Array(1, 3, 2, 4))
  }

  it should "transpose a 3D array correctly" in {
    val flattened = ArrayNd.fromArray(
      Array(0, 8, 16, 4, 12, 20, 1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3, 11, 19, 7, 15, 23)
    )
    transposeArr3d.transpose.flatten shouldEqual flattened
  }

}
