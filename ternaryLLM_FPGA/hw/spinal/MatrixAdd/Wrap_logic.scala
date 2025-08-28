package MatrixAdd

import gemmacc.coyote._
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4Config, AxiLite4SlaveFactory}
import spinal.lib._

class Wrap_logic (sysConf : ConfigSys) extends Component{

  val  io = new Bundle {

    val axi_ctrl = slave(AxiLite4(AxiLite4Config(64, 64)))

    // DESCRIPTORS
    val sq_rd = master Stream(ReqT())
    val sq_wr = master Stream(ReqT())
    val cq_rd = slave Stream(ack_t())
    val cq_wr = slave Stream(ack_t())


    // CARD DATA STREAMS
    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

  }

  // Create SlaveFactory
  val ctrlR = new AxiLite4SlaveFactory(io.axi_ctrl, useWriteStrobes = true)

  // Assign addresses to registers for axi_ctrl
  val input_start = ctrlR.createReadAndWrite(Bool(), 0x00, 0, documentation = "Start Signal")
  val done = ctrlR.createReadOnly(Bool(), 0x08, 0, documentation = "Done Signal")
  val input_base_addr_X = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x10, 0, documentation = "Base address X")
  val input_base_addr_W = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x18, 0, documentation = "Base address W")
  val input_base_addr_Y = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x20, 0, documentation = "Base address Y")

  // Initiate logic module
  val logic = new logic(sysConf)

  // Connect Signals to logic Module
  logic.io.start := input_start
  logic.io.base_addr_X := input_base_addr_X
  logic.io.base_addr_W := input_base_addr_W
  logic.io.base_addr_Y := input_base_addr_Y
  done := logic.io.done

  // Queues
  logic.io.sq_rd <> io.sq_rd
  logic.io.sq_wr <> io.sq_wr
  logic.io.cq_rd <> io.cq_rd
  logic.io.cq_wr <> io.cq_wr

  // Memory AXI
  logic.io.axis_card_recv <> io.axis_card_recv
  logic.io.axis_card_send <> io.axis_card_send

}
