package scalarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ArrayNdTest extends AnyFlatSpec with Matchers {

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

  it should "correctly print out a 2 by 2 matrix" in {
    val arr = ArrayNd.fill[Int](2, 2)(0)
    arr.toString shouldEqual
      s"""scalarray.ArrayNd(
         |  [[0, 0],
         |   [0, 0]]
         |)""".stripMargin
  }

}
