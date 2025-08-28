package gemmacc.sim.Op.coyote_v2

import gemmacc.coyote.AxiCoyote
import gemmacc.src.ConfigSys
import gemmacc.src.Op.coyote_v2.old.TopLevelOP_coyote_v2
import gemmacc.util.{AxiMemorySim, AxiMemorySimConfig, InputAndCheck}
import spinal.core.Component
import spinal.core.sim._

object TopLevelSim {

  val conf = new ConfigSys {}

  val tests = List(
    (4, 1 * conf.S_2, 256),
    (4, 2 * conf.S_2, 512),
    (4, 4 * conf.S_2, 1024),
    (8, 1 * conf.S_2, 512),
    (8, 2 * conf.S_2, 1024),
    (8, 4 * conf.S_2, 256),
    (16, 1 * conf.S_2, 1024),
    (16, 2 * conf.S_2, 256),
    (16, 4 * conf.S_2, 512)
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

    // addresswidtth : 64bit
    val base_addr_X = 0x0000000
    val base_addr_Y = 0x3000000
    val base_addr_W = 0x8000000
    val entries = ((0.5 * K) /2).toInt
    //val entries = 2
    val entries_per_K_slice = (entries / (K / config.K_slice)).toInt

    SimConfig.withWave.compile(new Component{
      val axiCoyote = new AxiCoyote(conf)
      val tGEMM = new TopLevelOP_coyote_v2(conf)

      // connect with AxiCoyote translater
      tGEMM.io.sq_rd <> axiCoyote.io.sq_rd
      tGEMM.io.sq_wr <> axiCoyote.io.sq_wr
      tGEMM.io.cq_rd <> axiCoyote.io.cq_rd
      tGEMM.io.cq_wr <> axiCoyote.io.cq_wr
      tGEMM.io.axis_card_recv <> axiCoyote.io.axis_card_recv
      tGEMM.io.axis_card_send <> axiCoyote.io.axis_card_send
      tGEMM.io.simPublic()
      axiCoyote.io.simPublic()

    } ).doSim { dut =>

      dut.clockDomain.forkStimulus(10)

      dut.tGEMM.io.M #= M
      dut.tGEMM.io.N #= N
      dut.tGEMM.io.K #= K
      dut.tGEMM.io.Non_zero_per_K_slice #= entries_per_K_slice
      dut.tGEMM.io.base_addr_X #= base_addr_X
      dut.tGEMM.io.base_addr_W #= base_addr_W
      dut.tGEMM.io.base_addr_Y #= base_addr_Y

      val axiMem = AxiMemorySim(dut.axiCoyote.io.mem, dut.clockDomain, axiConfig)
      axiMem.start()

      val X = InputAndCheck.generateX(M, K)
      axiMem.memory.writeArray(base_addr_X, X)

      val W = InputAndCheck.generateW(config.S, entries, N)
      axiMem.memory.writeArray(base_addr_W, W)

      dut.tGEMM.io.start #= true

      dut.clockDomain.waitSamplingWhere(dut.tGEMM.io.done.toBoolean)
      dut.tGEMM.io.start #= false

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
    println("======= GEMM Accelerator Simulation Completed =======")
  }
}
