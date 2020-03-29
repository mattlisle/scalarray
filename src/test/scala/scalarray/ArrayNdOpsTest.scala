package scalarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ArrayNdOpsTest extends AnyFlatSpec with Matchers {

  it should "map an array of one type to another" in {
    val arr = Array.fill(10)(0.5)
    val arrNd = ArrayNd.fill(10)(1)
    arrNd.map(_.toDouble / 2) shouldEqual ArrayNd.fromArray(arr)
  }

  it should "map a transposed array" in {
    val before = ArrayNd.fromArray((0 until 6).toArray).reshape(2, 3)
    val after = ArrayNd.fromArray((0 until 12 by 2).toArray).reshape(2, 3).transpose
    before.map(_ * 2).transpose shouldEqual after
  }

}
