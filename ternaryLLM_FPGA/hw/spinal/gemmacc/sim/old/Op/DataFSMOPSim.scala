package gemmacc.sim.Op

import gemmacc.src.AXI_version.DataFSM_OP_AXI
import gemmacc.src.ConfigSys
import gemmacc.util.{AxiMemorySim, AxiMemorySimConfig, InputAndCheck}
import spinal.core.sim._

object DataFSMOPSim {
  def main(args: Array[String]): Unit = {

    val config = new ConfigSys {}

    val M = 4  // Number of Rows of X & Y
   //val K_slice = 64 // Number of Columns of X and Rows of W
    val K = 256
    val N = 32// Number of Columns of W & Y
    //val S = 64 // S/2 must be < N
    val base_addr_X = 0x0000
    val base_addr_Y = 0x3000
    val base_addr_W = 0x8000
    val entries = 64
    val entries_per_Kslice = 0

    SimConfig.withWave.compile{
      val d = new DataFSM_OP_AXI(config)
      d.Buffer_X.simPublic()
      d
    }.doSim { dut =>
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
      axiMem.memory.writeArray(base_addr_X, X)

      val W = InputAndCheck.generateW(S = config.S, entries = entries, N = N)

      // Write W to Memory
      axiMem.memory.writeArray(base_addr_W, W)

      // Create Dummy Values for Y and test if it writes on to memory
      val Y = Array.tabulate(config.S_2) (i => 5.toShort)


      dut.io.start #= true

      dut.clockDomain.waitSampling(8000)

      for(i <- 0 until config.ENTRIES_PER_BEAT_X_Y){
      }





      dut.io.start #= false



    }
  }
}