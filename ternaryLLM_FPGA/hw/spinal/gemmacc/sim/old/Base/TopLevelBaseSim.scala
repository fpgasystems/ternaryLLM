package gemmacc.sim.Base

import gemmacc.src.Base.TopLevelBase
import gemmacc.src.ConfigSys
import gemmacc.util.{AxiMemorySim, AxiMemorySimConfig, InputAndCheck}
import spinal.core.sim.{SimConfig, _}

object TopLevelBaseSim {

  val conf = new ConfigSys {}

  val tests = List(
    (1, 2 * conf.S_2, 256),
    (1, 4 * conf.S_2, 512),
    (2, 2 * conf.S_2, 256),
    (2, 4 * conf.S_2, 1024),
    (4, 4 * conf.S_2 , 512),
    (4, 2 * conf.S_2, 1024)
  )

  def runTest(M: Int, N: Int, K: Int): Unit = {
    println(s"\n===== Starting Test: M=$M, N=$N, K=$K =====")

    val config = new ConfigSys {}

    val axiConfig = AxiMemorySimConfig(
      maxOutstandingReads = 8,
      maxOutstandingWrites = 8,
      readResponseDelay = 1,
      writeResponseDelay = 1
    )

    val base_addr_X = 0x0000
    val base_addr_Y = 0x3000
    val base_addr_W = 0x8000
    val entries = (0.5 * K/2).toInt
    //val entries = 2
    val entries_per_K_slice = (entries / (K / config.K_slice)).toInt

    SimConfig.withWave.compile(new TopLevelBase(config)).doSim { dut =>
      dut.io.simPublic()
      dut.clockDomain.forkStimulus(10)

      dut.io.M #= M
      dut.io.N #= N
      dut.io.K #= K
      dut.io.Non_zero_per_K_slice #= entries_per_K_slice
      dut.io.base_addr_X #= base_addr_X
      dut.io.base_addr_W #= base_addr_W
      dut.io.base_addr_Y #= base_addr_Y

      val axiMem = AxiMemorySim(dut.io.AXI, dut.clockDomain, axiConfig)
      axiMem.start()

      val X = InputAndCheck.generateX(M, K)
      axiMem.memory.writeArray(base_addr_X, X)

      val W = InputAndCheck.generateW(config.S, entries, N)
      axiMem.memory.writeArray(base_addr_W, W)

      dut.io.start #= true
      dut.clockDomain.waitSamplingWhere(dut.io.done.toBoolean)
      dut.io.start #= false

      val Y = InputAndCheck.naiveGEMM(M, N, K, config.S, entries, X, W).flatten

      println("Y Naive:")
      for (i <- 0 until M * N) {
        print(Y(i) + " ")
        if ((i + 1) % N == 0) println()
      }
      println()

      println("Y Calculated:")
      for (i <- 0 until M * N) {
        val byte0 = axiMem.memory.read(base_addr_Y + i * 2) & 0xFF
        val byte1 = axiMem.memory.read(base_addr_Y + i * 2 + 1) & 0xFF
        val unsignedValue = (byte1 << 8) | byte0
        val signedValue = if (unsignedValue >= 0x8000) unsignedValue - 0x10000 else unsignedValue
        assert(signedValue == Y(i), s"Value must be ${Y(i).toInt}, but was $signedValue")
        print(signedValue + " ")
        if ((i + 1) % N == 0) println()
      }

      println(s"===== Completed Test: M=$M, N=$N, K=$K =====\n")
    }
  }

  def main(args: Array[String]): Unit = {
    println("======= GEMM Accelerator Simulation Start =======")
  for ((m ,n, k) <- tests) {
   runTest(m, n, k)
}
   // runTest(4, 128, 512)
    println("======= GEMM Accelerator Simulation Completed =======")
  }
}
