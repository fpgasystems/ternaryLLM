package gemmacc.src.Op.coyote_v2

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.fsm.{EntryPoint, State, StateMachine}
import spinal.lib.{Counter, master, slave}

import scala.language.postfixOps

class DataFSM(conf: ConfigSys) extends Component{

  val io = new Bundle {
    val start = in Bool()
    val done = out Bool()

    // Coyote V2 Interface
    val sq_rd = master Stream (ReqT())
    val sq_wr = master Stream (ReqT())
    val cq_rd = slave Stream (ack_t())
    val cq_wr = slave Stream (ack_t())

    val axis_card_recv = slave(AXI4SR())

    // Runtime Inputs
    val M = in UInt (16 bits)
    val N = in UInt (16 bits)
    val K = in UInt (16 bits)
    val base_addr_X = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt (conf.axiConfig.addressWidth bits)
    val Non_zero_per_K_slice = in UInt (16 bits)
    val expected_beats_X = in UInt(16 bits)


    // Logic in and output
    val x = out Bits (conf.DATA_WIDTH bits) // Input 512-Bit X-values from RAM driven by DataFSM
    val w = master Stream Bits (conf.DATA_WIDTH bits) // Input Stream 512-Bit W-values driven by DataFSM
    val select_Y = out Bits(conf.UNROLL_M bits) // UNROLL_M selectors needed
    val enable_X = out Vec(Bits(8 bits),conf.UNROLL_M)
    val reset_acc = out Bool()
    val done_acc = in Bool()
    //Debug

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
  }

  val done = Reg(Bool()).init(False)
  io.done := done
  val enable_write = Bool()
  val addr_w = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))
  val addr_r = Reg(UInt(conf.ADDR_WIDTH_BUFFER bits))

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
  val buffer_out = Buffer_X.readAsync(addr_r)

  // Intialize Counter
  val cnt_N = Reg(UInt(16 bits))
  val cnt_M = Reg(UInt(16 bits))
  val cnt_K = Reg(UInt(16 bits))
  val cnt_entries = Counter(16 bits)
  val cnt_beats_W = Counter(32 bits) // tracks all beats for whole W
  val cnt_unroll_Y = Counter(conf.UNROLL_M)
  val cnt_unroll_rows = Counter(16 bits)
  val k_c = Counter(16 bit)
  val cnt_s = Counter(conf.TOTAL_BEATS_KSLICE)
  val cnt_row = Counter(conf.UNROLL_M)
  val cnt_N_write = Reg(UInt(16 bits))
  val cnt_beats_x = Counter(1024)


  //val expected_beats_X = Reg(UInt(16 bits))
 // expected_beats_X := (((io.K << conf.OFFSET_X ) >> conf.AXI_DATA_SIZE) - 1).resized

  // DEBUG
  io.cnt_N := cnt_N
  io.cnt_M := cnt_M
  io.cnt_K := cnt_K
  io.cnt_entries := cnt_entries.value
  io.cnt_beats_W := cnt_beats_W.value
  io.cnt_unroll_Y := cnt_unroll_Y.value
  io.cnt_unroll_rows := cnt_unroll_rows.value
  io.k_c := k_c.value
  io.cnt_s := cnt_s.value
  io.cnt_row := cnt_row.value
  io.cnt_N_write := cnt_N_write

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

  // Default io.w stream
  io.w.valid := False
  io.w.payload := io.axis_card_recv.tdata

  // Default io.x
  io.x := 0

  // Default io.select_Y
  io.select_Y := 0

  // Default io.enable (depending on UNROLL)
  io.enable_X.foreach(_ := B(0, 8 bits))

  // Default reset_acc
  io.reset_acc := False


  // FSM
  val dataFSM = new StateMachine {
    val IDLE = new State with EntryPoint
    val NEXT_ROWS, LOAD_X_Buffer, LOAD_X_DATA, LOAD_SLICE, READ_W_AR, LOAD_W_DATA, WAIT, MEM_ADDR_Y, MEM_WRITE_Y, SET_ADDR, WAIT_ACC = new State

    // IDLE: Wait for start signal
    IDLE.whenIsActive {

      // Reset done signal
      when(!io.start) {
        done := False
      }

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
      cnt_unroll_rows.clear()
      k_c.clear()
      cnt_s.clear()
      cnt_row.clear()
      cnt_N_write := 0
      cnt_beats_x.clear()

      io.reset_acc := True

      // once we receive start signal start loading
      when(io.start && !io.done) {
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

      when(cnt_M + conf.UNROLL_M >= io.M) { // we have finished computation
        done := True
        goto(IDLE)
      }.otherwise { // we go to next 4 rows
        io.reset_acc := True
        cnt_beats_x.clear() // reset beat_count for burst read
        goto(LOAD_X_Buffer)
      }
    }

    // READ_X_AR: set the configuration for AXI-ar && increment counter_X
    LOAD_X_Buffer.whenIsActive {

      // BURST_READ
      io.sq_rd.payload.vaddr := (io.base_addr_X + ((cnt_unroll_rows.value).resize(48 bits) * conf.UNROLL_M) * (io.K.resize(48 bits) << conf.BYTE_X)).resized
      io.sq_rd.payload.len := (io.K << conf.OFFSET_X).resized // Length in Bytes
      io.sq_rd.payload.last := True // Assuming it works like that
      io.sq_rd.valid := True


      cnt_beats_W.clear() // clear Weights

      when(io.sq_rd.fire) {
        goto(LOAD_X_DATA)
      }

    }


    // LOAD_X_DATA: read one row data and route to RAM, we need to stay here until we have put all data into buffer
    LOAD_X_DATA.whenIsActive {

      // complete Queue is ready and we are ready to receive data on the axis_card_recv
      io.cq_rd.ready := True
      io.axis_card_recv.tready := True

      addr_w := (cnt_row.value * conf.BUFFERSIZE_PER_ROW + k_c.value).resized

      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // We receive 512bits through the Bus
        enable_write := True

        k_c.increment
        cnt_beats_x.increment()

        // Assignment Overlap if we put this code above the when
        cnt_K := cnt_K + conf.ENTRIES_PER_BEAT_X_Y

        when(cnt_K >= (io.K - conf.ENTRIES_PER_BEAT_X_Y)) {
          k_c.clear()
          cnt_K := 0
          cnt_row.increment()
        }

        // Check if this is the last beat
        when(cnt_beats_x.value >= io.expected_beats_X) {
          cnt_row.clear()
          k_c.clear()
          cnt_K := 0
          cnt_s.clear()
          goto(SET_ADDR)
        }

      }

    }

    // NEED THIS SO WE HAVE NOT A DELAY OF 1 CYLE FOR READING BUFFER_OUT
    SET_ADDR.whenIsActive {
      addr_r := (cnt_row.value * conf.BUFFERSIZE_PER_ROW + k_c.value * conf.BUFFER_ENTRIES_PER_KSLICE + cnt_s.value).resized
      goto(LOAD_SLICE)
    }


    // put data into io.x and give enable_X signal
    LOAD_SLICE.whenIsActive {
      // store the value into it

      // READ until we have read 1 K_slice of one row and then go to the next row
      // cnt_row can be used to acccess right enable_X
      io.x := buffer_out

      // set the enable_X signal(ONE_HOT encoding)
      switch(cnt_s.value){
        is(0) { io.enable_X(cnt_row.value.resized) := B"00000001" }
        is(1) { io.enable_X(cnt_row.value.resized) := B"00000010" }
        is(2) { io.enable_X(cnt_row.value.resized) := B"00000100" }
        is(3) { io.enable_X(cnt_row.value.resized) := B"00001000" }
        is(4) { io.enable_X(cnt_row.value.resized) := B"00010000" }
        is(5) { io.enable_X(cnt_row.value.resized) := B"00100000" }
        is(6) { io.enable_X(cnt_row.value.resized) := B"01000000" }
        is(7) { io.enable_X(cnt_row.value.resized) := B"10000000" }
      }

      when(cnt_s.value >= conf.TOTAL_BEATS_KSLICE- 1 && cnt_row.value >= conf.UNROLL_M - 1) {
        cnt_s.clear()
        cnt_row.clear()
        cnt_K := cnt_K + conf.K_slice  // we have put first K_SCLICES into PE
        goto(READ_W_AR)
      } elsewhen (cnt_s.value >= conf.BUFFER_ENTRIES_PER_KSLICE -1) { // then we put 1 K_SLICE into PE
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

      io.sq_rd.payload.vaddr := (io.base_addr_W + (cnt_beats_W.value << conf.OFFSET)).resized
      io.sq_rd.payload.len := conf.Row_Data_W  // 64 Bytes = 64 Weights of 32 Columns
      io.sq_rd.payload.last := True
      io.sq_rd.valid := True

      when(io.sq_rd.fire) {
        goto(LOAD_W_DATA)
      }

    }

    // LOAD_W_DATA: read S indices and put into buffer , stay until we have put all data into the buffers
    LOAD_W_DATA.whenIsActive {

      io.cq_rd.ready := True
      io.axis_card_recv.tready := io.w.ready  // wait until ALL PE are ready to receive Weights

      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready) { // arrives 512 bit

        io.w.valid := True

        cnt_beats_W.increment() // for address calculation

        goto(WAIT_ACC)
      }

    }

    WAIT_ACC.whenIsActive{

      when(io.done_acc) {
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
            cnt_N := cnt_N + conf.S_2 //
            cnt_K := 0
            goto(WAIT)
          }

        }


      }

    }

    // TODO: Do we really need it?
    val waitCounter = Counter(3 bits) // Counter for waiting cyles
    io.waitCounter := waitCounter
    // Wait: wait for 4 cycles
    WAIT.whenIsActive {
      when(waitCounter.value === 7) {
        waitCounter.clear()
        goto(MEM_ADDR_Y)
      }.otherwise {
        waitCounter.increment()
      }
    }

    // MEM_ADDR_Y :  Configure AXI-AW Channel
    MEM_ADDR_Y.whenIsActive {

      io.sq_wr.payload.vaddr := (io.base_addr_Y + ((((cnt_unroll_rows.value * conf.UNROLL_M) + cnt_unroll_Y.value).resize(48 bits) * (io.N.resize(48 bits) << conf.BYTE_Y)) + (cnt_N_write.resize(48 bits) << conf.BYTE_Y))).resized
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

      // set the enable_X signal(ONE_HOT encoding)
      switch(cnt_unroll_Y.value){
        is(0) { io.select_Y := B"0001" }
        is(1) { io.select_Y := B"0010" }
        is(2) { io.select_Y := B"0100" }
        is(3) { io.select_Y := B"1000" }
      }

      // axis_send driven by TopLevel !!

      when(io.cq_wr.fire) { // received feedback


        when(cnt_unroll_Y >= conf.UNROLL_M - 1){

          // we have stored 4 times S/2 columns
          cnt_unroll_Y.clear()
          k_c.clear()

          when(cnt_N < io.N) { // if we have some weight columns left
            cnt_N_write := cnt_N_write + conf.S_2
            io.reset_acc := True // reset accumulators
            goto(SET_ADDR) // if we have columns of W left
          }.otherwise {
            cnt_N_write := 0
            goto(NEXT_ROWS)
          }

        }.otherwise{
          cnt_unroll_Y.increment()
          goto(MEM_ADDR_Y)
        }

      }

    }

  }


  dataFSM.postBuild{
    io.dataFSM_state := dataFSM.stateReg.asBits
  }




}
