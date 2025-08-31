package gemmacc.sim.testLogic

import gemmacc.coyote.AxiCoyote
import gemmacc.src.ConfigSys
import MatrixAdd._
import gemmacc.util.{AxiMemorySim, AxiMemorySimConfig}
import spinal.core.Component
import spinal.core.sim._

import java.nio.{ByteBuffer, ByteOrder}

object logic_sim {

  def main(args: Array[String]): Unit = {
    val conf = new ConfigSys {}

    val axiConfig = AxiMemorySimConfig(
      maxOutstandingReads = 8,
      maxOutstandingWrites = 8,
      readResponseDelay = 1,
      writeResponseDelay = 1
    )

    val base_addr_X = 0x1000
    val base_addr_Y = 0x3000
    val base_addr_W = 0x8000


    SimConfig.withWave.compile(new Component {
      val axiCoyote = new AxiCoyote(conf)
      val logicUnit = new logic(conf)

      // Connect logic streams to AxiCoyote streams
      logicUnit.io.sq_rd <> axiCoyote.io.sq_rd
      logicUnit.io.sq_wr <> axiCoyote.io.sq_wr
      logicUnit.io.cq_rd <> axiCoyote.io.cq_rd
      logicUnit.io.cq_wr <> axiCoyote.io.cq_wr
      logicUnit.io.axis_card_recv <> axiCoyote.io.axis_card_recv
      logicUnit.io.axis_card_send <> axiCoyote.io.axis_card_send

      logicUnit.io.simPublic()
      axiCoyote.io.simPublic()

    }).doSim { dut =>


      dut.clockDomain.forkStimulus(10)

      val axiMem = AxiMemorySim(dut.axiCoyote.io.mem, dut.clockDomain, axiConfig)
      axiMem.start()

      val buffer_X = ByteBuffer.allocate(20) // 20 Bytes per short
      buffer_X .order(ByteOrder.LITTLE_ENDIAN)
      val X = Array.tabulate(10)(i => 1.toShort) // all values with 1
      X.foreach(i => buffer_X.putShort(i))
      axiMem.memory.writeArray(base_addr_X, buffer_X.array())

      val buffer_W = ByteBuffer.allocate(20) // 20 Bytes per short
      buffer_W.order(ByteOrder.LITTLE_ENDIAN)
      val W = Array.tabulate(10)(i => i.toShort)
      W.foreach(i => buffer_W.putShort(i))
      axiMem.memory.writeArray(base_addr_W, buffer_W.array())

      dut.logicUnit.io.start #= false
      dut.logicUnit.io.base_addr_X #= base_addr_X
      dut.logicUnit.io.base_addr_W #= base_addr_W
      dut.logicUnit.io.base_addr_Y #= base_addr_Y

      dut.clockDomain.waitSampling(20)

      dut.logicUnit.io.start #= true

      dut.clockDomain.waitSamplingWhere(dut.logicUnit.io.done.toBoolean)

      dut.logicUnit.io.start #= false

      println("Y Calculated:")
      for (i <- 0 until 10) {
        val byte0 = axiMem.memory.read(base_addr_Y + i * 2) & 0xFF
        val byte1 = axiMem.memory.read(base_addr_Y + i * 2 + 1) & 0xFF
        val unsignedValue = (byte1 << 8) | byte0
        val signedValue = if (unsignedValue >= 0x8000) unsignedValue - 0x10000 else unsignedValue
        print(signedValue + " ")
      }

    }
  }
}
