package gemmacc.src.design

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4Config, AxiLite4SlaveFactory}
import spinal.lib.{master, slave}

class WrapGEMM(sysConf : ConfigSys) extends Component{

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

  val TopLevel = new TopLevel(sysConf)

  // Queues
  TopLevel.io.sq_rd <> io.sq_rd
  TopLevel.io.sq_wr <> io.sq_wr
  TopLevel.io.cq_rd <> io.cq_rd
  TopLevel.io.cq_wr <> io.cq_wr

  // Memory AXI
  TopLevel.io.axis_card_recv <> io.axis_card_recv
  TopLevel.io.axis_card_send <> io.axis_card_send

  val ctrlR = new AxiLite4SlaveFactory(io.axi_ctrl, useWriteStrobes = true)

  // Create Read and Write register
  val input_M = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x00, 0, documentation = "Input M")
  val input_N = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x08, 0, documentation =  "Input N")
  val input_K = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x10, 0, documentation = "Input K")
  val input_Nz_values_Kslice = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x18, 0, documentation = "How many entries per K_Slice")
  val input_start = ctrlR.createReadAndWrite(Bool(), 0x20, 0, documentation = "Start Signal")
  val input_base_addr_X = ctrlR.createReadAndWrite(UInt(sysConf.VADDR_BITS bits), 0x28, 0, documentation = "Base address X")
  val input_base_addr_W = ctrlR.createReadAndWrite(UInt(sysConf.VADDR_BITS bits), 0x30, 0, documentation = "Base address W")
  val input_base_addr_Y = ctrlR.createReadAndWrite(UInt(sysConf.VADDR_BITS bits), 0x38, 0, documentation = "Base address Y")
  val finished = ctrlR.createReadOnly(Bool(), 0x40, 0, documentation = "Done Signal")
  val expected_beats      = ctrlR.createReadAndWrite(UInt(sysConf.BIT_WIDTH_INPUT bits), 0x48, 0, documentation = "Input Burst Read beats")
  val cnt_cyles = ctrlR.createReadOnly(Bits(40 bits), 0x50, 0, documentation = "Counter for Cycles")

  // Connect Register to io of TopLevel
  TopLevel.io.M := input_M
  TopLevel.io.N:= input_N
  TopLevel.io.K := input_K
  TopLevel.io.Non_zero_per_K_slice := input_Nz_values_Kslice
  TopLevel.io.start := input_start
  TopLevel.io.base_addr_X := input_base_addr_X
  TopLevel.io.base_addr_W := input_base_addr_W
  TopLevel.io.base_addr_Y := input_base_addr_Y
  finished := TopLevel.io.done
  TopLevel.io.expected_beats_X := expected_beats
  cnt_cyles := TopLevel.io.cnt_cycles

  //DEBUG
//  val reg_cnt_N           = ctrlR.createReadOnly(UInt(16 bits),                   0x48, 0, documentation = "Debug: cnt_N")
//  val reg_cnt_M           = ctrlR.createReadOnly(UInt(16 bits),                   0x50, 0, documentation = "Debug: cnt_M")
//  val reg_cnt_K           = ctrlR.createReadOnly(UInt(16 bits),                   0x58, 0, documentation = "Debug: cnt_K")
//  val reg_cnt_entries     = ctrlR.createReadOnly(UInt(16 bits),                   0x60, 0, documentation = "Debug: cnt_entries")
//  val reg_cnt_beats_W     = ctrlR.createReadOnly(UInt(16 bits),                   0x68, 0, documentation = "Debug: cnt_beats_W (all beats for W)")
//  val reg_cnt_unroll_Y    = ctrlR.createReadOnly(UInt(log2Up(sysConf.UNROLL_M) bits),0x70, 0, documentation = "Debug: cnt_unroll_Y")
//  val reg_cnt_unroll_rows = ctrlR.createReadOnly(UInt(16 bits),                   0x78, 0, documentation = "Debug: cnt_unroll_rows")
//  val reg_k_c             = ctrlR.createReadOnly(UInt(16 bits),                   0x80, 0, documentation = "Debug: k_c")
//  val reg_cnt_s           = ctrlR.createReadOnly(UInt(log2Up(sysConf.TOTAL_BEATS_KSLICE) bits), 0x88, 0, documentation = "Debug: cnt_s")
//  val reg_cnt_row         = ctrlR.createReadOnly(UInt(log2Up(sysConf.UNROLL_M) bits),0x90, 0, documentation = "Debug: cnt_row")
//  val reg_cnt_N_write     = ctrlR.createReadOnly(UInt(16 bits),                   0x98, 0, documentation = "Debug: cnt_N_write")
//  val reg_waitCounter     = ctrlR.createReadOnly(UInt(3 bits),                    0xA0, 0, documentation = "Debug: waitCounter")
//  val reg_dataFSM_state   = ctrlR.createReadOnly(Bits(4 bits),                    0xA8, 0, documentation = "Debug: dataFSM_state")
//  val reg_cnt_beats_y     = ctrlR.createReadOnly(UInt(16 bits),                   0xB0, 0, documentation = "Debug: cnt_beats_y")

  // Assignments of debug registers
//  reg_cnt_N           := TopLevel.io.cnt_N
//  reg_cnt_M           := TopLevel.io.cnt_M
//  reg_cnt_K           := TopLevel.io.cnt_K
//  reg_cnt_entries     := TopLevel.io.cnt_entries
//  reg_cnt_beats_W     := TopLevel.io.cnt_beats_W
//  reg_cnt_unroll_Y    := TopLevel.io.cnt_unroll_Y
//  reg_cnt_unroll_rows := TopLevel.io.cnt_unroll_rows
//  reg_k_c             := TopLevel.io.k_c
//  reg_cnt_s           := TopLevel.io.cnt_s
//  reg_cnt_row         := TopLevel.io.cnt_row
//  reg_cnt_N_write     := TopLevel.io.cnt_N_write
//  reg_waitCounter     := TopLevel.io.waitCounter
//  reg_dataFSM_state   := TopLevel.io.dataFSM_state
//  reg_cnt_beats_y     := TopLevel.io.cnt_beats_y




}
