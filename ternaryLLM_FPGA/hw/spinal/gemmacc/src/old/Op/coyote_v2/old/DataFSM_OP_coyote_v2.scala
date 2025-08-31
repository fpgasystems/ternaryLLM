
package gemmacc.src.Op.coyote_v2.old

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

import scala.language.postfixOps

class DataFSM_OP_coyote_v2(conf: ConfigSys) extends Component {

  val io = new Bundle {
    val start = in Bool()

    // Coyote V2 Interface
    val sq_rd = master Stream (ReqT())
    val sq_wr = master Stream (ReqT())
    val cq_rd = slave Stream (ack_t())
    val cq_wr = slave Stream (ack_t())

    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

    // Runtime Inputs
    val M = in UInt (16 bits)
    val N = in UInt (16 bits)
    val K = in UInt (16 bits)
    val base_addr_X = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt (conf.axiConfig.addressWidth bits)
    val Non_zero_per_K_slice = in UInt (16 bits)

    // Logic in and output
    val x = out Vec(Vec(SInt(conf.BIT_WIDTH_X_Y bits), conf.K_slice), conf.UNROLL_M)
    val kernel = Vec(Vec(master Stream UInt(conf.BIT_WIDTH_INDEX bits), conf.S), conf.UNROLL_M)
    val result_GEMM = in Vec(Vec(SInt(conf.BIT_WIDTH_X_Y bits), conf.S_2), conf.UNROLL_M)
    val reset_acc = out Bool()
    val done = out Bool()
  }
  val done = Reg(Bool()).init(False)
  io.done := done
  val valid = Bool()
  val enable_write = Bool()
  val addr_w = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))
  val addr_r = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))

  // Set enable_write to false as default
  enable_write := False
  valid := False

  // Creates a Buffer of Size conf.BUFFERSIZE depending on Model
  val Buffer_X = Mem(Bits(conf.axiConfig.dataWidth bits), conf.BUFFERSIZE)


  // Write Port
  Buffer_X.write(
    address = addr_w,
    data = io.axis_card_recv.tdata,
    enable = enable_write
  )

  // TODO: Asynch read triggered bin for initialize Buffer_X
  // Read Port, Vec(SInt(BITWIDTH_X bits) , ENTRIES_PER_BEAT)
  val Buffer_Out = Buffer_X.readAsync(addr_r).subdivideIn(conf.ENTRIES_PER_BEAT_X_Y slices).map(_.asSInt)


  // Intialize Counter
  val cnt_N = Reg(UInt(16 bits))
  val cnt_M = Reg(UInt(16 bits))
  val cnt_K = Reg(UInt(16 bits))
  val cnt_w = Counter(16 bits)
  val cnt_entries = Counter(16 bits)
  val cnt_beats_W = Counter(32 bits) // tracks all beats for whole W
  val cnt_beats_y = Counter(16 bits, io.axis_card_send.tready && io.axis_card_send.tvalid)
  val cnt_unroll = Counter(16 bits)
  val cnt_unroll_rows = Counter(16 bits)
  val row_c = Counter(conf.UNROLL_M)
  val k_c = Counter(16 bit)
  val cnt_s = Counter(16 bits)
  val cnt_row = Counter(conf.UNROLL_M)
  val cnt_N_write = Reg(UInt(16 bits))

  // Default Configurations:
  // sq_rd
  io.sq_rd.valid := False
  io.sq_rd.payload := ReqT().getZero // dest = 0 , opcode = conf.LOCAL_READ = 0 , strm = conf.STRM_CARD = 0

  // sq_wr
  io.sq_wr.valid := False
  io.sq_wr.payload := ReqT().getZero

  // cq_rd
  io.cq_rd.ready := False
  // cq_wr
  io.cq_wr.ready := False

  // axi_card_recv
  io.axis_card_recv.tready := False

  // axi_card_send
  io.axis_card_send.tdata := 0
  io.axis_card_send.tkeep := 0
  io.axis_card_send.tid := 0
  io.axis_card_send.tlast := False
  io.axis_card_send.tvalid := False


  io.reset_acc := False


  // Default Value for io.kernel
  for (i <- 0 until conf.UNROLL_M) {
    io.kernel(i).foreach(_.valid := valid)
    io.kernel(i).foreach(_.payload := 0)
  }

  val x_reg = Vec(Reg(Vec(SInt(conf.BIT_WIDTH_X_Y bits), conf.K_slice)), conf.UNROLL_M)

  io.x := x_reg

  // Assign values to kernel stream // CAUSING that it only works with BITWIDTH of 8 bits !!
  for (i <- 0 until conf.ENTRIES_PER_BEAT_W) {
    for (j <- 0 until conf.UNROLL_M) {
      io.kernel(j)((cnt_w.value * conf.ENTRIES_PER_BEAT_W + i).resize(conf.BITWIDTH_S)).payload := io.axis_card_recv.tdata.subdivideIn(conf.ENTRIES_PER_BEAT_W slices)(i).asUInt
    }
  }

  // FSM
  val dataFSM = new StateMachine {
    val IDLE = new State with EntryPoint
    val NEXT_ROWS, LOAD_X_Buffer, LOAD_X_DATA, LOAD_SLICE, READ_W_AR, LOAD_W_DATA, WAIT, MEM_ADDR_Y, MEM_WRITE_Y, SET_ADDR = new State

    // IDLE: Wait for start signal
    IDLE.whenIsActive {

      // Reset done signal
      when(!io.start) {
        io.done := False;
      }

      // Set values for reading/writing of Buffer in IDLE State
      enable_write := False
      addr_r := 0
      addr_w := 0
      // TODO: Check if this is really needed, simulation works (Causes an binary File and then: ERROR: [Route 35-2] Design is not legally routed. There are 952412 node overlaps. )
      Buffer_X.init(Vec.fill(conf.BUFFERSIZE)(B(0, conf.axiConfig.dataWidth bits)))

      // reset counters
      cnt_N := 0
      cnt_M := 0
      cnt_K := 0
      cnt_w.clear()
      cnt_entries.clear()
      cnt_beats_W.clear()
      cnt_beats_y.clear()
      cnt_unroll.clear()
      cnt_unroll_rows.clear()
      row_c.clear()
      k_c.clear()
      cnt_s.clear()
      cnt_row.clear()
      cnt_N_write := 0

      // once we receive start signal start loading
      when(io.start && !done) {
        goto(LOAD_X_Buffer)
      }
    }


    NEXT_ROWS.whenIsActive {
      // Add Unroll_M factor to overall
      cnt_M := cnt_M + conf.UNROLL_M

      cnt_unroll_rows.increment()
      cnt_K := 0
      cnt_N := 0
      k_c.clear()
      cnt_s.clear()
      cnt_row.clear()

      when(cnt_M + conf.UNROLL_M >= io.M) {
        done := True
        goto(IDLE)
      }.otherwise {
        io.reset_acc := True
        goto(LOAD_X_Buffer)
      }
    }

    // READ_X_AR: set the configuration for AXI-ar && increment counter_X

    // TODO: adjust address + len for whole UNROLL Rows, OPTIMIZASTION with shifting
    LOAD_X_Buffer.whenIsActive {

      io.sq_rd.payload.vaddr := (io.base_addr_X + (cnt_unroll_rows.value * conf.UNROLL_M * (io.K << conf.BYTE_X))).resized
      io.sq_rd.payload.len := (conf.UNROLL_M * io.K * conf.DATA_SIZE_X_Y_BYTE).resized // Length in Bytes       // Change for local Simulation
      io.sq_rd.payload.last := True // Assuming it works like that
      io.sq_rd.valid := True


      cnt_beats_W.clear() // clear Weights

      when(io.sq_rd.fire) {
        goto(LOAD_X_DATA)
      }

    }


    // LOAD_X_DATA: read one row data and route to buffer, we need to stay here until we have put all data into buffer
    LOAD_X_DATA.whenIsActive {

      // complete Queue is ready and we are ready to receive data on the axis_card_recv
      io.cq_rd.ready := True
      io.axis_card_recv.tready := True

      addr_w := (row_c.value * conf.BUFFERSIZE_PER_ROW + k_c.value).resized

      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // We receive 512bits through the Bus
        enable_write := True

        k_c.increment

        // Assignment Overlap if we put this code above the when
        cnt_K := cnt_K + conf.ENTRIES_PER_BEAT_X_Y

        when(cnt_K >= (io.K - conf.ENTRIES_PER_BEAT_X_Y)) {
          k_c.clear()
          cnt_K := 0
          row_c.increment()
        }

      }
      // Check if this is the last beat
      when(io.axis_card_recv.tlast) {
        row_c.clear()
        k_c.clear()
        cnt_K := 0
        cnt_s.clear()
        goto(SET_ADDR)
      }

    }

    // NEED THIS SO WE HAVE NOT A DELAY OF 1 CYLE FOR READING BUFFER_OUT
    SET_ADDR.whenIsActive {
      addr_r := (cnt_row.value * conf.BUFFERSIZE_PER_ROW + k_c.value * conf.BUFFER_ENTRIES_PER_KSLICE + cnt_s.value).resized
      goto(LOAD_SLICE)
    }


    // Store data into io.x
    LOAD_SLICE.whenIsActive {
      // store the value into it

      for (l <- 0 until conf.ENTRIES_PER_BEAT_X_Y) {
        x_reg(cnt_row.value)((cnt_s.value * conf.ENTRIES_PER_BEAT_X_Y + l).resized) := Buffer_Out(l)
      }

      when(cnt_s.value >= conf.BUFFER_ENTRIES_PER_KSLICE - 1 && cnt_row.value >= conf.UNROLL_M - 1) {
        cnt_s.clear()
        cnt_row.clear()
        goto(READ_W_AR)
      } elsewhen (cnt_s.value >= conf.BUFFER_ENTRIES_PER_KSLICE - 1) {
        cnt_s.clear()
        cnt_row.increment()
        goto(SET_ADDR)
      } otherwise {
        cnt_s.increment()
        goto(SET_ADDR)
      }

    }

    //Format : MERGED TCS Grouped (Uniformal)
    // READ_W_AR: set configuation for AXI-ar
    READ_W_AR.whenIsActive {

      io.sq_rd.payload.vaddr := (io.base_addr_W + (cnt_beats_W << conf.OFFSET)).resized
      io.sq_rd.payload.len := conf.Row_Data_W
      io.sq_rd.payload.last := True
      io.sq_rd.valid := True

      when(io.sq_rd.fire) {
        goto(LOAD_W_DATA)
      }

    }


    // LOAD_W_DATA: read S indices and put into buffer , stay until we have put all data into the buffers
    LOAD_W_DATA.whenIsActive {

      io.cq_rd.ready := True
      io.axis_card_recv.tready := True

      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // arrives 512 bit

        valid := True

        cnt_beats_W.increment() // for address calculation

      }

      when(io.axis_card_recv.tlast) {
        cnt_w.clear()
        when(cnt_entries < io.Non_zero_per_K_slice - 1) { // read Non_zero_per_K_slice times S indexes for S/2 columns
          cnt_entries.increment()
          goto(READ_W_AR)
        }.otherwise {
          // we have read all entries for this K_slice
          cnt_entries.clear()
          cnt_K := cnt_K + conf.K_slice
          when(cnt_K + conf.K_slice < io.K) { // we have some slices left
            k_c.increment()
            goto(SET_ADDR)
          }.otherwise {
            // we are done with S/2 columns of W
            cnt_N := cnt_N + conf.S_2 // we have calculate values for cn
            cnt_K := 0
            goto(WAIT)
          }

        }

      }.otherwise {
      cnt_w.increment()
    }
    }

    val waitCounter = Counter(3 bits) // Counter for waiting cyles
    // Wait: wait for 4 cycles
    WAIT.whenIsActive {
      when(waitCounter.value === 3) {
        waitCounter.clear()
        goto(MEM_ADDR_Y)
      }.otherwise {
        waitCounter.increment()
      }
    }


    // MEM_ADDR_Y :  Configure AXI-AW Channel
    MEM_ADDR_Y.whenIsActive {

      io.sq_wr.payload.vaddr := (io.base_addr_Y + ((((cnt_unroll_rows.value * conf.UNROLL_M) + cnt_unroll.value) * (io.N << conf.BYTE_Y)) + (cnt_N_write << conf.BYTE_Y))).resized
      io.sq_wr.payload.len := conf.Row_Data_Y
      io.sq_wr.payload.last := True
      io.sq_wr.valid := True

      when(io.sq_wr.fire) {
        goto(MEM_WRITE_Y)
      }


    }
    // MEM_WRITE: put data into w.data buffer
    MEM_WRITE_Y.whenIsActive {

      io.cq_wr.ready := True

      io.axis_card_send.tdata := (io.result_GEMM((cnt_unroll.value).resize(log2Up(conf.UNROLL_M))).reverse).map(_.asBits).reduce(_ ## _).subdivideIn(conf.axiConfig.dataWidth bits)(cnt_beats_y.resized)
      io.axis_card_send.tkeep.setAll()
      io.axis_card_send.tlast := cnt_beats_y.value === conf.total_numb_beat_Y
      io.axis_card_send.tvalid := True

      when(io.cq_wr.fire) { // received feedback

        cnt_beats_y.clear()

        when(cnt_unroll.value < conf.UNROLL_M - 1) { // we need to loop back to write the other values into memory
          cnt_unroll.increment()
          goto(MEM_ADDR_Y)
        }.otherwise {
          // we have stored 4 times S/2 columns
          cnt_unroll.clear()
          k_c.clear()
          when(cnt_N < io.N) { // if we have some kernels left
            cnt_N_write := cnt_N_write + conf.S_2
            io.reset_acc := True
            goto(SET_ADDR) // if we have columns of W left
          }.otherwise {
            cnt_N_write := 0
            goto(NEXT_ROWS)
          }
        }

      }

    }

  }
}
