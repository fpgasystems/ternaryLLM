package gemmacc.src.Op.coyote_v2.old

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4Config, AxiLite4SlaveFactory}
import spinal.lib.{master, slave}

class WrapSys(sysConf : ConfigSys) extends Component{

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

  val TopLevel = new TopLevelOP_coyote_v2(sysConf)
  val ctrlR = new AxiLite4SlaveFactory(io.axi_ctrl, useWriteStrobes = true)


  // Create Read and Write register
  val input_M = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x00, 0, documentation = "Input M")
  val input_N = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x08, 0, documentation =  "Input N")
  val input_K = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x10, 0, documentation = "Input K")
  val input_Nz_values_Kslice = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x18, 0, documentation = "How many entries per K_Slice")
  val input_start = ctrlR.createReadAndWrite(Bool(), 0x20, 0, documentation = "Start Signal")
  val input_base_addr_X = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x28, 0, documentation = "Base address X")
  val input_base_addr_W = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x30, 0, documentation = "Base address W")
  val input_base_addr_Y = ctrlR.createReadAndWrite(UInt(sysConf.axiConfig.addressWidth bits), 0x38, 0, documentation = "Base address Y")
  val finished = ctrlR.createReadOnly(Bool(), 0x40, 0, documentation = "Done Signal")

  // Connect Register to io of TopLevel
  TopLevel.io.M := input_M
  TopLevel.io.N := input_N
  TopLevel.io.K := input_K
  TopLevel.io.Non_zero_per_K_slice := input_Nz_values_Kslice
  TopLevel.io.start := input_start
  TopLevel.io.base_addr_X := input_base_addr_X
  TopLevel.io.base_addr_W := input_base_addr_W
  TopLevel.io.base_addr_Y := input_base_addr_Y
  finished := TopLevel.io.done

  // Queues
  TopLevel.io.sq_rd <> io.sq_rd
  TopLevel.io.sq_wr <> io.sq_wr
  TopLevel.io.cq_rd <> io.cq_rd
  TopLevel.io.cq_wr <> io.cq_wr

  // Memory AXI
  TopLevel.io.axis_card_recv <> io.axis_card_recv
  TopLevel.io.axis_card_send <> io.axis_card_send

}
