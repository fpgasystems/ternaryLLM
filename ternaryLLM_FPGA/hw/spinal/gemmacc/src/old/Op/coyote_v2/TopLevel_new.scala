package gemmacc.src.Op.coyote_v2

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.core.{Bundle, Component}
import spinal.lib.{Counter, master, slave}

import scala.language.postfixOps

class TopLevel_new(conf: ConfigSys) extends Component {

  val  io = new Bundle {

    val start = in Bool()
    val done = out Bool()
    val M = in UInt(16 bits)
    val N = in UInt(16 bits)
    val K = in UInt(16 bits)
    val Non_zero_per_K_slice = in UInt(16 bits)
    val base_addr_X = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt(conf.axiConfig.addressWidth bits)

    // DESCRIPTORS
    val sq_rd = master Stream(ReqT())
    val sq_wr = master Stream(ReqT())
    val cq_rd = slave Stream(ack_t())
    val cq_wr = slave Stream(ack_t())


    // CARD DATA STREAMS
    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

    val cnt_N = out UInt(16 bits)
    val cnt_M = out UInt(16 bits)
    val cnt_K = out UInt(16 bits)
    val cnt_entries = out UInt(16 bits)
    val cnt_beats_W = out UInt (32 bits) // tracks all beats for whole W
    val cnt_unroll_Y = out UInt (log2Up(conf.UNROLL_M) bits)
    val cnt_unroll_rows = out UInt(16 bits)
    val k_c = out UInt (16 bit)
    val cnt_s = out UInt (log2Up(conf.TOTAL_BEATS_KSLICE)bits)
    val cnt_row = out UInt (log2Up(conf.UNROLL_M) bits)
    val cnt_N_write = out UInt(16 bits)
    val waitCounter = out UInt (3 bits)
    val dataFSM_state = out Bits(4 bits)
    val cnt_beats_y = out UInt(16 bits)
  }

  val FSM = new DataFSM(conf)
  val PE = Seq.tabulate(conf.UNROLL_M)(i => new PE(conf))

  val cnt_beats_y = Counter(16 bits, io.axis_card_send.tready && io.axis_card_send.tvalid)


  // DEBUG
  io.cnt_N           := FSM.io.cnt_N
  io.cnt_M           := FSM.io.cnt_M
  io.cnt_K           := FSM.io.cnt_K
  io.cnt_entries     := FSM.io.cnt_entries
  io.cnt_beats_W     := FSM.io.cnt_beats_W
  io.cnt_unroll_Y    := FSM.io.cnt_unroll_Y
  io.cnt_unroll_rows := FSM.io.cnt_unroll_rows
  io.k_c             := FSM.io.k_c
  io.cnt_s           := FSM.io.cnt_s
  io.cnt_row         := FSM.io.cnt_row
  io.cnt_N_write     := FSM.io.cnt_N_write
  io.waitCounter     := FSM.io.waitCounter
  io.dataFSM_state   := FSM.io.dataFSM_state
  io.cnt_beats_y     := cnt_beats_y.value


  // Connect Runtime Inputs
  FSM.io.M <> io.M
  FSM.io.N <> io.N
  FSM.io.K <> io.K
  FSM.io.base_addr_X <> io.base_addr_X
  FSM.io.base_addr_W <> io.base_addr_W
  FSM.io.base_addr_Y <> io.base_addr_Y
  FSM.io.Non_zero_per_K_slice <> io.Non_zero_per_K_slice

  // Connect start and done signals
  io.start <> FSM.io.start
  io.done <> FSM.io.done

  //Connect Coyote_v2 to FSM
  io.sq_rd <> FSM.io.sq_rd
  io.sq_wr <> FSM.io.sq_wr
  io.cq_rd <> FSM.io.cq_rd
  io.cq_wr <> FSM.io.cq_wr

 // io.axis_card_send <> FSM.io.axis_card_send
  io.axis_card_recv <> FSM.io.axis_card_recv

  // Connect PE to FSM
  for(i <- 0 until conf.UNROLL_M){
    PE(i).io.x <> FSM.io.x
    PE(i).io.w.valid := FSM.io.w.valid
    PE(i).io.w.payload := FSM.io.w.payload
    PE(i).io.enable_X <> FSM.io.enable_X(i)
    PE(i).io.reset_acc <> FSM.io.reset_acc
  }

  // For ready signal we need to and the ready signals COULD be a problem
  FSM.io.w.ready := PE.map(_.io.w.ready).reduce(_ && _)
  FSM.io.done_acc := PE.map(_.io.done_acc).reduce(_ && _)


  io.axis_card_send.tkeep.setAll()
  io.axis_card_send.tid := 0
  io.axis_card_send.tlast := cnt_beats_y.value === conf.total_numb_beat_Y

  when(io.cq_wr.fire){
    cnt_beats_y.clear()
  }

  // Switch for selecting right output to write back
  switch(FSM.io.select_Y){
    is(B"0001"){
      io.axis_card_send.tdata := PE(0).io.y
      io.axis_card_send.tvalid := True
    }

    is(B"0010"){
      io.axis_card_send.tdata := PE(1).io.y
      io.axis_card_send.tvalid := True
    }

    is(B"0100"){
      io.axis_card_send.tdata := PE(2).io.y
      io.axis_card_send.tvalid := True
    }

    is(B"1000"){
      io.axis_card_send.tdata := PE(3).io.y
      io.axis_card_send.tvalid := True
    }

    default{
      io.axis_card_send.tdata := 0
      io.axis_card_send.tvalid := False
    }

  }



}
