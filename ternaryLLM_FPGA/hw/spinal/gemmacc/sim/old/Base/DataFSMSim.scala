package gemmacc.sim.Base

import gemmacc.src.Base.DataFSM_Base
import gemmacc.src.ConfigSys
import gemmacc.util.{AxiMemorySim, AxiMemorySimConfig, InputAndCheck}
import spinal.core.sim._

object DataFSMSim {
  def main(args: Array[String]): Unit = {

    val config = new ConfigSys {}

    val M = 1  // Number of Rows of X & Y
   //val K_slice = 64 // Number of Columns of X and Rows of W
    val K = 64
    val N = 32// Number of Columns of W & Y
    //val S = 64 // S/2 must be < N
    val base_addr_X = 0x0000
    val base_addr_Y = 0x1000
    val base_addr_W = 0x2000
    val entries = 64
    val entries_per_Kslice = 0

    SimConfig.withWave.compile(new DataFSM_Base(config)).doSim { dut =>
      // Make all IO visible in waveform
      dut.io.simPublic()
      // Create Clock
      dut.clockDomain.forkStimulus(10)

      dut.io.M  #= M
      dut.io.N #= N
      dut.io.K #= K
      dut.io.Non_zero_per_K_slice #= entries// not used here
      dut.io.base_addr_X #= base_addr_X
      dut.io.base_addr_W #=  base_addr_W
      dut.io.base_addr_Y #= base_addr_Y


      val axiConfig = AxiMemorySimConfig(
        maxOutstandingReads = 8,
        maxOutstandingWrites = 8,
        readResponseDelay = 1,
        writeResponseDelay = 1
      )

      val axiMem = AxiMemorySim(dut.io.AXI, dut.clockDomain,axiConfig)
      axiMem.start()

      val X = InputAndCheck.generateX(M,config.K_slice)
//
//      for(i <- 0 until M * K){
//        println(X(i).toInt)
//      }
      // Write X to Memory
      axiMem.memory.writeArray(base_addr_X, X)

      val W = InputAndCheck.generateW(S = config.S, entries = entries, N = N)

      // Write W to Memory
      axiMem.memory.writeArray(base_addr_W, W)

      // Create Dummy Values for Y and test if it writes on to memory
      val Y = Array.tabulate(config.S_2) (i => 5.toShort)

      for(i <- 0 until config.S_2){
        dut.io.result_GEMM(i) #= Y(i)
      }

      dut.io.start #= true

        dut.clockDomain.waitSampling(10000)


//      print("Y : ")
//      for(i <- 0 until S/2){
//        print(axiMem.memory.read(0x1040 + i * 2))
//      }
//      println()



      dut.io.start #= false



    }
  }
}