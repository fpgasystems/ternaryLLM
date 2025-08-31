package gemmacc.sim.Base

import gemmacc.src.Base.TernaryGEMM
import spinal.core._
import spinal.core.sim._

import scala.collection.mutable.ListBuffer

object TernaryGEMMSim {
  def main(args: Array[String]): Unit = {
    val S = 4  // Must be even
    val S_2 = S /2    // Length of Vec of Stream for output
    val bitwidth = 32  // bitwidth
    val testValues = List(10,5,6,3)  // Test values for input buffer

    // For second input
    val second_input_plus = 2
    val second_input_minus = 1

    SimConfig.withWave.compile(new TernaryGEMM(S, bitwidth)).doSim { dut =>


      // ListBuffer for expectedValue
      val expectedValue = ListBuffer.fill(S/2)(0)

      // Clock setup
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.start #= true
      dut.clockDomain.waitSampling()
      dut.io.start #= false

      // Put Input into Buffer first Round
      for (i <- 0 until S) {
        dut.io.buffer(i).valid #= true
        dut.io.buffer(i).payload #= testValues(i)
      }
      dut.clockDomain.waitSampling(1)
      //dut.io.out_valid #= true

      for (i <- 0 until S) {
        dut.io.buffer(i).valid #= false
      }
      dut.clockDomain.waitSampling(4)
     // dut.io.accumulate #= true



// Debug:
//      for(i <- 0 until S ){
//        println(s" First Buffer Acc_value ${i}: ${dut.io.buffer(i).payload.toInt}")
//      }
//      for(i <- 0 until S ) {
//        println(s" First Round Acc_value ${i}: ${dut.accDebug(i).toInt}")
//      }

    //dut.io.accumulate #= false
//      // Put Input into Buffer second Round with accumulator Bool
//    for (i <- 0 until S) {
//       dut.io.buffer(i).valid #= true
//      if(i % 2 == 0){
//         dut.io.buffer(i).payload #= second_input_plus
//       } else {
//          dut.io.buffer(i).payload #= second_input_minus
//       }
//
//     }
//      dut.clockDomain.waitSampling(2)
//
//     // dut.io.accumulate #= true

      //dut.clockDomain.waitSampling(2)

//      for (i <- 0 until S) {
//        println(s" Second Round Buffer_value ${i}: ${dut.io.buffer(i).payload.toInt}")
//      }
//
//
//      for(i <- 0 until S ){
//        println(s" Second Round Acc_value ${i}: ${dut.accDebug(i).toInt}")
//      }

      // With Second Round:
      //dut.io.out_valid #= true

      // calculate expected outputs
      for(i <- 0 until S_2) {
        // with second Round:
//        expectedValue(i) = (testValues(i * 2) + second_input_plus) - (testValues(2 * i + 1) + second_input_minus)
        expectedValue(i) = testValues(i * 2)  - testValues(2 * i + 1)
      }

      // check outputs
      for(i <- 0 until S_2) {
        val expected =  expectedValue(i).toInt
        val output_val = dut.io.output(i).toInt
        println(s"Output ${i}: ${output_val} (expected: ${expected})")
        assert(output_val == expected, s"Output Value = ${output_val} but expected value ${expected}")
      }

    }
  }
}