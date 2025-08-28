package gemmacc.sim.Base

import gemmacc.src.Base.{ActivationBuffer, TernaryGEMM}
import gemmacc.util.InputAndCheck
import spinal.core.Component
import spinal.core.sim._


class ActivationToGEMM(S: Int, K_slice: Int , index_bitwidth: Int, bitwidth: Int ) extends Component{

  val activationBuffer = new ActivationBuffer(S, K_slice, index_bitwidth, bitwidth)
  val ternaryGEMM = new TernaryGEMM(S, bitwidth)

  // make them accessible in Simulation
  activationBuffer.io.simPublic()
  ternaryGEMM.io.simPublic()

  // Connect Output Buffer of ActivationBuffer with Input Buffer from TernaryGEMM
  for (i <- 0 until S) {
    ternaryGEMM.io.buffer(i) << activationBuffer.io.output(i)
  }


}

object ActivationToTernaryGEMMSim {
  def main(args: Array[String]): Unit = {
    val S = 64 // must be even
    val K_slice = 128
    val index_bitwidth = 16
    val bitwidth = 16
    val S_2 = S/2

    val X = InputAndCheck.generateX(1,K_slice) // Input testdata with length K_slice
    val kernel_idx = InputAndCheck.generateW(S,1,S) // Test indices


    SimConfig.withWave.compile(new ActivationToGEMM(S, K_slice, index_bitwidth, bitwidth)).doSim { dut =>

      dut.clockDomain.forkStimulus(period = 10)

    // Initialize X of ActivationBuffer
      for(i <- 0 until K_slice ) {
        dut.activationBuffer.io.x(i) #= X(i)
      }

      // Initialize signals for ActivationBuffer
      for(i <- 0 until S ) {
        dut.activationBuffer.io.kernel(i).valid #= false
        dut.activationBuffer.io.kernel(i).ready #= true
      }



      dut.ternaryGEMM.io.start #= true
      dut.clockDomain.waitSampling()
      dut.ternaryGEMM.io.start #= false



      for(i <- 0 until S){

        // simulate kernel index as input of activationbuffer
        dut.activationBuffer.io.kernel(i).valid #=true
        dut.activationBuffer.io.kernel(i).payload #= kernel_idx(i)

        // Wait for FSM to accept Index
        dut.clockDomain.waitSamplingWhere(dut.activationBuffer.io.kernel(i).ready.toBoolean)

        // Wait for output data
        dut.clockDomain.waitSamplingWhere(dut.activationBuffer.io.output(i).valid.toBoolean)
        dut.activationBuffer.io.kernel(i).valid #= false

        //dut.activationBuffer.io.output(i).ready #= true

        println(s"Buffer TernaryGEMM ${i}: val = ${dut.ternaryGEMM.io.buffer(i).payload.toInt}, ready = ${dut.ternaryGEMM.io.buffer(i).ready.toBoolean} , valid = ${dut.ternaryGEMM.io.buffer(i).valid.toBoolean}")
      }

      //dut.ternaryGEMM.io.out_valid #= true

      for (i <- 0 until S) {
        dut.ternaryGEMM.io.buffer(i).valid #= false
      }

      // dut.ternaryGEMM.io.accumulate #= true

      dut.clockDomain.waitSampling()

      for(i <- 0 until S_2){
        println(s"Output ${i}: ${dut.ternaryGEMM.io.output(i).toInt}")
      }


    }


  }
}
