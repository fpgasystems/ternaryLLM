package gemmacc.sim.Base

import gemmacc.src.Base.ActivationBuffer
import spinal.core.sim._



object ActivationBufferSim {
  def main(args: Array[String]): Unit = {
    val S = 4 // Must be even
    val K_slice = 6 // Length of Vec of Stream for output
    val bitwidth = 32 // bitwidth
    val idx_bitwidth = 3
    val X = List(0,1,2,3,4,5) // Input testdata with length K_slice
    val kernel_idx = List(2,1,4,5,6,0) // Test indices with length S

    SimConfig.withWave.compile(new ActivationBuffer(S,K_slice, idx_bitwidth, bitwidth)).doSim { dut =>

      //Initiate Clock
      dut.clockDomain.forkStimulus(period = 10)

      // Initiate X
      for(i <- 0 until K_slice ) {
        dut.io.x(i) #= X(i)
      }

      // Initiate signal
      for(i <- 0 until S ){
        dut.io.kernel(i).valid #= false
      }

      for(i <- 0 until S){


        dut.io.kernel(i).payload #= kernel_idx(i)
        dut.io.kernel(i).valid #=true

        // Wait for FSM to accept the index
        dut.clockDomain.waitSamplingWhere(dut.io.kernel(i).ready.toBoolean)


        // Wait for output data
        dut.clockDomain.waitSamplingWhere(dut.io.output(i).valid.toBoolean)
        dut.io.kernel(i).valid #= false

        //Verify output
        val expected = X(kernel_idx(i))
        println(s"Output ${i}: ${dut.io.output(i).payload.toInt} (expected: ${expected})")

        dut.io.output(i).ready #= true

      }

      }






    }

  }

