name := "scalarray"

version := "0.1"

scalaVersion := "2.13.1"

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/releases"

libraryDependencies += "com.storm-enroute" % "scalameter_2.13" % "0.19"
libraryDependencies += "org.scalatest" % "scalatest_2.13" % "3.1.1" % "test"
libraryDependencies += "org.mockito" % "mockito-core" % "1.9.0" % "test"

testFrameworks += new TestFramework("org.scalameter.ScalaMeterFramework")

logBuffered := false
parallelExecution in Test := false
