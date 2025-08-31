package gemmacc.src.design

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.{Counter, master, slave}

import scala.language.postfixOps

// if it don't pass through: int8

class TopLevel(conf: ConfigSys) extends Component {

  val  io = new Bundle {

    // DESCRIPTORS
    val sq_rd = master Stream(ReqT())
    val sq_wr = master Stream(ReqT())
    val cq_rd = slave Stream(ack_t())
    val cq_wr = slave Stream(ack_t())


    // CARD DATA STREAMS
    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

    val start = in Bool()
    val done = out Bool()
    val M = in UInt(16 bits)
    val N = in UInt(16 bits)
    val K = in UInt(16 bits)
    val Non_zero_per_K_slice = in UInt(16 bits)
    val expected_beats_X = in UInt(16 bits)
    val base_addr_X = in UInt (conf.VADDR_BITS bits)
    val base_addr_W = in UInt (conf.VADDR_BITS bits)
    val base_addr_Y = in UInt (conf.VADDR_BITS bits)

    // Count Cycles
    val cnt_cycles = out Bits(40 bits)

    // Debug
//    val cnt_N = out UInt(16 bits)
//    val cnt_M = out UInt(16 bits)
//    val cnt_K = out UInt(16 bits)
//    val cnt_entries = out UInt(16 bits)
//    val cnt_beats_W = out UInt (16 bits) // tracks all beats for whole W
//    val cnt_unroll_Y = out UInt (log2Up(conf.UNROLL_M) bits)
//    val cnt_unroll_rows = out UInt(16 bits)
//    val k_c = out UInt (16 bit)
//    val cnt_s = out UInt (log2Up(conf.TOTAL_BEATS_KSLICE)bits)
//    val cnt_row = out UInt (log2Up(conf.UNROLL_M) bits)
//    val cnt_N_write = out UInt(16 bits)
//    val waitCounter = out UInt (3 bits)
//    val dataFSM_state = out Bits(4 bits)
//    val cnt_beats_y = out UInt(16 bits)
  }

  val FSM = new DataFSM(conf)

  //Connect Coyote_v2 to FSM
  io.sq_rd <> FSM.io.sq_rd
  io.sq_wr <> FSM.io.sq_wr
  io.cq_rd <> FSM.io.cq_rd
  io.cq_wr <> FSM.io.cq_wr

  // io.axis_card_send <> FSM.io.axis_card_send
  io.axis_card_recv <> FSM.io.axis_card_recv

  // Connect Runtime Inputs
  FSM.io.M <> io.M
  FSM.io.N <> io.N
  FSM.io.K <> io.K
  FSM.io.base_addr_X <> io.base_addr_X
  FSM.io.base_addr_W <> io.base_addr_W
  FSM.io.base_addr_Y <> io.base_addr_Y
  FSM.io.Non_zero_per_K_slice <> io.Non_zero_per_K_slice
  FSM.io.expected_beats_X <> io.expected_beats_X
  FSM.io.cnt_cycles <> io.cnt_cycles

  // Connect start and done signals
  io.start <> FSM.io.start
  io.done <> FSM.io.done


  val PE_Array = Array.fill(conf.UNROLL_M)(new PE(conf))
  val yVec = Vec(PE_Array.map(_.io.y))

  // Connect PE to FSM
  for(i <- 0 until conf.UNROLL_M){
    PE_Array(i).io.x <> FSM.io.x
    PE_Array(i).io.w.valid := FSM.io.w.valid
    PE_Array(i).io.w.payload := FSM.io.w.payload
    PE_Array(i).io.enable_X <> FSM.io.enable_X(i)
    PE_Array(i).io.reset_acc <> FSM.io.reset_acc
  }

  FSM.io.w.ready := True

  io.axis_card_send.tvalid := FSM.io.enable_Y_write
  io.axis_card_send.tdata := yVec(FSM.io.select_Y)

  val cnt_beats_y = Counter(10 bits, io.axis_card_send.tready && io.axis_card_send.tvalid)
  io.axis_card_send.tkeep.setAll()
  io.axis_card_send.tid := 0
  io.axis_card_send.tlast := cnt_beats_y.value === conf.total_numb_beat_Y

  // DEBUG
//  io.cnt_N           := FSM.io.cnt_N
//  io.cnt_M           := FSM.io.cnt_M
//  io.cnt_K           := FSM.io.cnt_K
//  io.cnt_entries     := FSM.io.cnt_entries
//  io.cnt_beats_W     := FSM.io.cnt_beats_W
//  io.cnt_unroll_Y    := FSM.io.cnt_unroll_Y
//  io.cnt_unroll_rows := FSM.io.cnt_unroll_x
//  io.k_c             := FSM.io.k_c
//  io.cnt_s           := FSM.io.cnt_s
//  io.cnt_row         := FSM.io.cnt_row
//  io.cnt_N_write     := FSM.io.cnt_N_write
//  io.waitCounter     := FSM.io.waitCounter
//  io.dataFSM_state   := FSM.io.dataFSM_state
//  io.cnt_beats_y     := cnt_beats_y.value




  when(io.cq_wr.fire){
    cnt_beats_y.clear()
  }



}
