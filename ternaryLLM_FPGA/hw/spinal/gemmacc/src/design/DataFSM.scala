  package gemmacc.src.design

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.fsm.{EntryPoint, State, StateMachine}
import spinal.lib.{Counter, Shift, master, slave}

import scala.language.postfixOps

class DataFSM(conf: ConfigSys) extends Component {

  val io = new Bundle {
    // Coyote V2 Interface
    val sq_rd = master Stream (ReqT())
    val sq_wr = master Stream (ReqT())
    val cq_rd = slave Stream (ack_t())
    val cq_wr = slave Stream (ack_t())

    val axis_card_recv = slave(AXI4SR())
    val x = out Bits (conf.DATA_WIDTH bits) // Input 512-Bit X-values from RAM driven by DataFSM
    val w = master Stream Bits(conf.DATA_WIDTH bits) // Input Stream 512-Bit W-values driven by DataFSM

    // Runtime Inputs
    val M = in UInt (16 bits)
    val N = in UInt (16 bits)
    val K = in UInt (16 bits)
    val base_addr_X = in UInt (conf.VADDR_BITS bits)
    val base_addr_W = in UInt (conf.VADDR_BITS bits)
    val base_addr_Y = in UInt (conf.VADDR_BITS bits)
    val Non_zero_per_K_slice = in UInt (16 bits)
    val expected_beats_X = in UInt (16 bits)

    // Logic in and output
    val select_Y = out UInt (conf.SHIFT_UNROLL bits) // UNROLL_M selectors needed
    val enable_X = out Vec(Bits(2 bits), conf.UNROLL_M)
    val reset_acc = out Bool()
    val cnt_cycles = out Bits (40 bits)
    val enable_Y_write = out Bool()
    //Debug
    //    val cnt_N = out UInt(16 bits)
    //    val cnt_M = out UInt(16 bits)
    //    val cnt_K = out UInt(16 bits)
    //    val cnt_entries = out UInt(16 bits)
    //    val cnt_beats_W = out UInt (16 bits) // tracks all beats for whole W
    //    val cnt_unroll_Y = out UInt (log2Up(conf.UNROLL_M) bits)
    //    val cnt_unroll_x = out UInt(16 bits)
    //    val k_c = out UInt (16 bit)
    //    val cnt_s = out UInt (log2Up(conf.TOTAL_BEATS_KSLICE)bits)
    //    val cnt_row = out UInt (log2Up(conf.UNROLL_M) bits)
    //    val cnt_N_write = out UInt(16 bits)
    //    val waitCounter = out UInt (3 bits)
    //    val dataFSM_state = out Bits(4 bits)
    val start = in Bool()
    val done = out Bool()

  }

  // Default io.w stream
  io.w.valid := False
  io.w.payload := io.axis_card_recv.tdata

  // sq_rd
  io.sq_rd.valid := False
  io.sq_rd.payload := ReqT().getZero // dest = 0 , opcode = conf.LOCAL_READ = 0 , strm = conf.STRM_CARD = 0

  io.sq_wr.valid := False
  io.sq_wr.payload := ReqT().getZero

  // cq_rd
  io.cq_rd.ready := False
  // cq_wr
  io.cq_wr.ready := False

  // axi_card_recv
  io.axis_card_recv.tready := False

  // Default io.enable (depending on UNROLL)
  io.enable_X.foreach(_ := B(0, 2 bits))

  val count_time_valid = Reg(Bool())
  val done = Reg(Bool()).init(False)
  io.done := done
  val enable_write = Bool()
  val addr_w = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))
  val addr_r = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))
  val got_remainder_read = Reg(Bool())

  // Set enable_write to false as default
  enable_write := False

  // Creates a RAM of Size conf.BUFFERSIZE depending on Model
  val Buffer_X = Mem(Bits(conf.axiConfig.dataWidth bits), conf.BUFFERSIZE)

  // Write Port
  Buffer_X.write(
    address = addr_w,
    data = io.axis_card_recv.tdata,
    enable = enable_write
  )

  // Read Port, 512-bits
  io.x := Buffer_X.readAsync(addr_r)

  // Intialize Counter
  val cnt_N = Reg(UInt(16 bits))
  val cnt_M = Reg(UInt(16 bits))
  val cnt_K = Reg(UInt(16 bits))
  val cnt_entries = Counter(16 bits)
  val cnt_beats_W = Counter(16 bits) // tracks all beats for whole W
  val cnt_unroll_Y = Counter(conf.UNROLL_M)
  val cnt_unroll_x = Counter(16 bits)
  val k_c = Counter(16 bit)
  val cnt_s = Counter(conf.TOTAL_BEATS_X)
  val cnt_row = Counter(conf.UNROLL_M)
  val cnt_N_write = Reg(UInt(16 bits))
  val cnt_beats_x = Counter(16 bits)
  val cnt_time = Counter(40 bits)
  val cnt_beats_slice = Counter(16 bits)
  val cnt_read_4K = Counter(16 bits)

  // Logic for 4K reading:
  val X_4rows_base = (cnt_unroll_x.value << conf.SHIFT_UNROLL) * io.K
  val X_1row_base = cnt_row.value * io.K // for 1 Row index
  val X_4k_1row_base = (X_4rows_base + X_1row_base) << conf.BYTE_X
  val X_4k_base = cnt_read_4K << conf.SHIFT_4K
  val x_load_addr = io.base_addr_X + X_4k_1row_base + X_4k_base

  // Calculations for Writing
  val base_unroll_Y_base = (cnt_unroll_x.value << conf.SHIFT_UNROLL) + cnt_unroll_Y.value
  val base_Y_row = base_unroll_Y_base * (io.N << conf.BYTE_Y)
  val base_Y_column = (cnt_N_write << conf.BYTE_Y)

  // Calculations for 4K reading and remainder logic
  val K_in_Bytes = io.K << conf.BYTE_X
  val total_4K_slice = K_in_Bytes >> log2Up(4096) // Number of 4K reads per row"
  val remainder_length = K_in_Bytes(11 downto 0)
  val got_all_4k = cnt_read_4K === total_4K_slice
  when(remainder_length === 0) {
    got_remainder_read := True
  }

  // Calculation Reading out of Mem
  val base_row_mem = (cnt_row.value << conf.BUFFERSIZE_PER_ROW)
  val base_Kslice_mem = (k_c.value << conf.SHIFT_BUFFER_ENTRIES_PER_KSLICE)

  // Logic for counting Cylces
  io.cnt_cycles := cnt_time.asBits
  when(count_time_valid)(cnt_time.increment())

  val beats_4K = Reg(UInt(16 bits))


  //  // DEBUG
  //  io.cnt_N := cnt_N
  //  io.cnt_M := cnt_M
  //  io.cnt_K := cnt_K
  //  io.cnt_entries := cnt_entries.value
  //  io.cnt_beats_W := cnt_beats_W.value
  //  io.cnt_unroll_Y := cnt_unroll_Y.value
  //  io.cnt_unroll_x := cnt_unroll_x.value
  //  io.k_c := k_c.value
  //  io.cnt_s := cnt_s.value
  //  io.cnt_row := cnt_row.value
  //  io.cnt_N_write := cnt_N_write


  // Default io.enable_Y_write
  io.enable_Y_write := False

  // Default io.select_Y
  io.select_Y := 0



  // Default reset_acc
  io.reset_acc := False


  // FSM
  val dataFSM = new StateMachine {
    val IDLE = new State with EntryPoint
    val NEXT_4ROWS, LOAD_X_4K, LOAD_X_DATA, LOAD_SLICE, READ_W_AR, LOAD_W_DATA, WAIT, MEM_ADDR_Y, MEM_WRITE_Y, SET_ADDR, WAIT_ACC = new State

    val waitCounter = Counter(3 bits) // Counter for waiting cyles

    // IDLE: Wait for start signal
    IDLE.whenIsActive {
      // Set values for reading/writing of Buffer in IDLE State
      enable_write := False
      addr_r := 0
      addr_w := 0
      //Causes an binary File(load it manually to vivado project)
      Buffer_X.init(Vec.fill(conf.BUFFERSIZE)(B(0, conf.axiConfig.dataWidth bits)))

      // reset counters
      cnt_N := 0
      cnt_M := 0
      cnt_K := 0
      cnt_entries.clear()
      cnt_beats_W.clear()
      cnt_unroll_Y.clear()
      cnt_unroll_x.clear()
      k_c.clear()
      cnt_s.clear()
      cnt_row.clear()
      cnt_N_write := 0
      cnt_beats_x.clear()
      cnt_read_4K.clear()
      cnt_beats_slice.clear()

      got_remainder_read := False
      io.reset_acc := True

      // Clock Counter
      count_time_valid.clear()

      when(!io.start) {
        done := False
      }
      // once we receive start signal start loading
      when(io.start && !io.done) {
        count_time_valid.set()
        cnt_time.clear()
        goto(LOAD_X_4K)
      }
    }


    // Checks if computation is completed
    NEXT_4ROWS.whenIsActive {
      cnt_unroll_x.increment()
      cnt_K := 0
      cnt_N := 0
      k_c.clear()
      cnt_s.clear()
      cnt_row.clear()


      when(cnt_M >= io.M) { // we have finished computation
        done := True
        goto(IDLE)
      }.otherwise { // we go to next 4 rows
        io.reset_acc := True
        cnt_beats_x.clear() // reset beat_count for burst read
        cnt_beats_W.clear() // clear Weights
        goto(LOAD_X_4K)
      }
    }

    // Set Configurations for reading 4K or remainder
    LOAD_X_4K.whenIsActive {

      io.sq_rd.payload.vaddr := x_load_addr.resized
      // 4k length
      io.sq_rd.payload.len := conf.four_K
      beats_4K := conf.EXPECTED_BEATS_4K

      when(got_all_4k) {
        // remainder
        io.sq_rd.payload.len := remainder_length.resized
        beats_4K := ((remainder_length >> conf.AXI_DATA_SIZE) - 1).resized
        io.sq_rd.payload.last := True
        io.sq_rd.valid := True

        when(io.sq_rd.fire) {
          when(!got_all_4k) {
            cnt_read_4K.increment()
          } otherwise {
            got_remainder_read.set()
          }
          goto(LOAD_X_DATA)
        }


      }


      // LOAD_X_DATA: read one row activations and store it into BRAM, we need to stay here until we have put all data into buffer
      LOAD_X_DATA.whenIsActive {

        // complete Queue is ready and we are ready to receive data on the axis_card_recv
        io.cq_rd.ready := True
        io.axis_card_recv.tready := True

        addr_w := ((cnt_row.value << conf.SHIFT_BUFFERSIZE_PER_ROW) + k_c.value).resized

        when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // We receive 512bits through the Bus

          enable_write := True

          k_c.increment
          cnt_beats_x.increment()
          cnt_beats_slice.increment()


          // Check if this is the last beat
          when(cnt_beats_x >= io.expected_beats_X) {
            cnt_row.clear()
            k_c.clear()
            cnt_K := 0
            cnt_s.clear()
            cnt_read_4K.clear()
            got_remainder_read := False
            cnt_beats_slice.clear()
            goto(SET_ADDR)
          } elsewhen (cnt_beats_slice >= beats_4K) {
            // if one_row has been read
            cnt_beats_slice.clear()
            when(got_all_4k && got_remainder_read) {
              cnt_row.increment()
              k_c.clear()
              cnt_read_4K.clear()
              got_remainder_read := False
            }
            goto(LOAD_X_4K)

          }


        }

      }
      // NEED THIS SO WE HAVE NOT A DELAY OF 1 CYCLE FOR READING BUFFER_OUT
      SET_ADDR.whenIsActive {
        addr_r := (base_row_mem + base_Kslice_mem + cnt_s.value).resized
        goto(LOAD_SLICE)
      }


      LOAD_SLICE.whenIsActive {

        // set the enable_X signal(ONE_HOT encoding)
        switch(cnt_s.value) {
          is(0) {
            io.enable_X(cnt_row.value.resized) := B"01"
          }
          is(1) {
            io.enable_X(cnt_row.value.resized) := B"10"
          }
        }

        when(cnt_s.value >= conf.TOTAL_BEATS_KSLICE_0 && cnt_row.value >= conf.UNROLL_M_0_INDEX) {
          cnt_s.clear()
          cnt_row.clear()
          cnt_K := cnt_K + conf.K_slice // we have put first K_Sclices into PE
          goto(READ_W_AR)
        } elsewhen (cnt_s.value >= conf.TOTAL_BEATS_KSLICE_0) { // then we put 1 K_SLICE into PE
          cnt_s.clear()
          cnt_row.increment()
          goto(SET_ADDR)
        } otherwise {
          cnt_s.increment()
          goto(SET_ADDR)
        }

      }

      // READ_W_AR: Set configuration for fetching weights.
      READ_W_AR.whenIsActive {

        io.sq_rd.payload.vaddr := (io.base_addr_W + (cnt_beats_W.value << conf.OFFSET)).resized
        io.sq_rd.payload.len := conf.Row_Data_W
        io.sq_rd.payload.last := True
        io.sq_rd.valid := True

        when(io.sq_rd.fire) {
          goto(LOAD_W_DATA)
        }

      }

      // LOAD_W_DATA: read S indices and put into io.w
      LOAD_W_DATA.whenIsActive {

        io.cq_rd.ready := True
        io.axis_card_recv.tready := io.w.ready

        when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // arrives 512 bit

          io.w.valid := True

          cnt_beats_W.increment() // for address calculation
          goto(WAIT_ACC)

        }

      }

      WAIT_ACC.whenIsActive {
        when(cnt_entries.value < io.Non_zero_per_K_slice - 1) { // read Non_zero_per_K_slice times S indexes for S/2 columns
          cnt_entries.increment()
          goto(READ_W_AR)
        }.otherwise {
          // we have read all entries for this K_slice
          cnt_entries.clear()

          when(cnt_K < io.K) { // we have some slices left
            k_c.increment()
            goto(SET_ADDR)
          }.otherwise {
            // we are done with S/2=32 columns of W
            cnt_N := cnt_N + conf.S_2
            cnt_K := 0
            goto(WAIT)
          }

        }

      }

      WAIT.whenIsActive {
        when(waitCounter.value === 7) {
          waitCounter.clear()
          goto(MEM_ADDR_Y)
        }.otherwise {
          waitCounter.increment()
        }
      }

      // MEM_ADDR_Y : Configure AXI write-queue
      MEM_ADDR_Y.whenIsActive {

        io.sq_wr.payload.vaddr := (io.base_addr_Y + base_Y_row + base_Y_column).resized
        io.sq_wr.payload.len := conf.Row_Data_Y
        io.sq_wr.payload.last := True
        io.sq_wr.valid := True

        when(io.sq_wr.fire) {
          goto(MEM_WRITE_Y)
        }


      }
      // MEM_WRITE: Sets the payload and waits until the write transaction completes.
      MEM_WRITE_Y.whenIsActive {

        io.cq_wr.ready := True

        io.enable_Y_write := True
        io.select_Y := cnt_unroll_Y.value

        when(io.cq_wr.fire) { // received feedback


          when(cnt_unroll_Y >= conf.UNROLL_M_0_INDEX) {

            // we have stored Unroll_M times S/2 columns
            cnt_unroll_Y.clear()
            k_c.clear()

            when(cnt_N < io.N) { // if we have some weight columns left
              cnt_N_write := cnt_N_write + conf.S_2
              io.reset_acc := True // reset accumulators
              goto(SET_ADDR)
            }.otherwise {
              cnt_N_write := 0
              cnt_M := cnt_M + conf.UNROLL_M
              goto(NEXT_4ROWS)
            }

          }.otherwise {
            cnt_unroll_Y.increment()
            goto(MEM_ADDR_Y)
          }

        }

      }

    }

    // Debug
    //  dataFSM.postBuild{
    //    io.dataFSM_state := dataFSM.stateReg.asBits
    //  }


  }
}
