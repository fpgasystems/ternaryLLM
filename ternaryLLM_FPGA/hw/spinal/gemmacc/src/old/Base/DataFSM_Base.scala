package gemmacc.src.Base

import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.fsm._

import scala.language.postfixOps

// TODO: clean this version

// M, N, K given at runtime
class DataFSM_Base(conf : ConfigSys) extends Component {

  val io = new Bundle {
    // ADD M,K,N, base_addr, non-zero-values
    val start = in Bool()
    val AXI = master(Axi4(conf.axiConfig))
    val M = in UInt(16 bits)
    val N = in UInt(16 bits)
    val K = in UInt(16 bits)
    val base_addr_X = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt(conf.axiConfig.addressWidth bits)
    val Non_zero_per_K_slice = in UInt(16 bits)
    val x = out Vec(SInt(conf.BIT_WIDTH_X_Y bits), conf.K_slice)
    val kernel = Vec(master Stream UInt(conf.BIT_WIDTH_INDEX bits), conf.S)
    val result_GEMM = in Vec(SInt(conf.BIT_WIDTH_X_Y bits), conf.S_2)
    val reset_acc = out Bool()
    val done = out Bool()
  }

  // Intialize Counter
  val cnt_N = Reg(UInt(16 bits))
  val cnt_M = Counter(16 bits)
  val cnt_K = Reg(UInt(16 bits))// At the moment we read whole Row of X
  val cnt_w = Counter(16 bits)
  val cnt_entries = Counter(16 bits)
  val cnt_beats_X = Counter(32 bits) // tracks all beats for whole X
  val cnt_beats_Y = Counter(32 bits) // tracks all beats for whole Y
  val cnt_beats_W = Counter(32 bits) // tracks all beats for whole W
  val cnt_slice = Counter(16 bits)

  // Done Signal
  val done = Reg(Bool()).init(False)
  io.done := done

  // Default Configurations AR
  io.AXI.ar.addr := 0
  io.AXI.ar.id := 0
  io.AXI.ar.len := 0
  io.AXI.ar.size := 0
  io.AXI.ar.setBurstINCR()
  io.AXI.ar.valid := False

  // Default Configuration for R
  io.AXI.r.ready := False

  // Default Configuration AW
  io.AXI.aw.addr := 0
  io.AXI.aw.id := 0
  io.AXI.aw.len := 0
  io.AXI.aw.size := 0
  io.AXI.aw.setBurstINCR()
  io.AXI.aw.valid := False

  // Default Configuration W
  io.AXI.w.data := 0
  io.AXI.w.last := False
  io.AXI.w.payload.strb.setAll()
  io.AXI.w.valid := False
  io.AXI.b.ready := False

  val update_x = Vec(Bool(), conf.total_numb_beat_X)
  io.reset_acc := False

  // Default Value for io.kernel
  io.kernel.foreach(_.valid := False)
  io.kernel.foreach(_.payload := 0)


  // Assign values to kernel stream
  for(i <- 0 until conf.ENTRIES_PER_BEAT_W) {
    io.kernel( (cnt_w.value * conf.ENTRIES_PER_BEAT_W + i).resize(conf.BITWIDTH_S)).payload := io.AXI.r.data.subdivideIn(conf.ENTRIES_PER_BEAT_W slices)(i).asUInt
  }

  // stores into write correct position of io.x
  for (j <- 0 until conf.total_numb_beat_X) {
    update_x(j) := False
    for (i <- 0 until conf.ENTRIES_PER_BEAT_X_Y) {
      io.x(j * conf.ENTRIES_PER_BEAT_X_Y + i) := RegNextWhen(io.AXI.r.data.subdivideIn(conf.ENTRIES_PER_BEAT_X_Y slices)(i), update_x(j) && cnt_M.value <= io.M ).asSInt
    }
  }


  // FSM
  val dataFSM = new StateMachine {
    val IDLE = new State with EntryPoint
    val NEXT_ROW, READ_X_AR, LOAD_X_DATA, READ_W_AR, LOAD_W_DATA, WAIT, MEM_ADDR_Y, MEM_WRITE_Y = new State

    // IDLE: Wait for start signal
    IDLE.whenIsActive {
      // reset counters
      cnt_N := 0
      cnt_M.clear()
      cnt_K := 0
      cnt_slice.clear()
      cnt_beats_X.clear()
      cnt_beats_Y.clear()
      cnt_beats_W.clear()
      cnt_entries.clear()
      // once we receive start signal start loading
      when(io.start && !done) {
        goto(READ_X_AR)
      }
    }

    NEXT_ROW.whenIsActive{
      cnt_M.increment()

      cnt_slice.clear()
      cnt_K := 0

      when(cnt_M.value >  io.M  ) {
        done:= True
        goto(IDLE)
      }.otherwise{
        goto(READ_X_AR)
      }

    }


    // READ_X_AR: set the configuration for AXI-ar && increment counter_X
    READ_X_AR.whenIsActive {
      // TODO: this version read multiple times same k_slice
      // AXI-AR Configurations
      io.AXI.ar.addr := (io.base_addr_X  + ((cnt_M.value) * ( io.K << conf.BYTE_X) ) + (cnt_slice.value  * (conf.total_numb_beat_X << conf.OFFSET))).resize(conf.axiConfig.addressWidth)
      io.AXI.ar.len := conf.total_numb_beat_X - 1
      io.AXI.ar.size := conf.AXI_DATA_SIZE
      io.AXI.ar.valid := True

      cnt_beats_W.clear() // clear Weights

      when(io.AXI.ar.fire) {
        goto(LOAD_X_DATA)
      }

    }

    // Counter for Beat
    val cnt_beat_x = Counter(conf.total_numb_beat_X)

    // LOAD_X_DATA: read one row data and route to buffer, we need to stay here until we have put all data into buffer
    LOAD_X_DATA.whenIsActive {

      io.AXI.r.ready := True // We are ready to receive data

      when(io.AXI.r.fire) { // We receive 512bits through the Bus

        update_x(cnt_beat_x.value) := True // the update of x is now valid

        when(io.AXI.r.last) {
          cnt_beat_x.clear()
        }.otherwise {
          cnt_beat_x.increment()
        }
      }

      // Check if this is the last beat
      when(io.AXI.r.last) {
        cnt_K := cnt_K + conf.K_slice
        goto(READ_W_AR)
      }

    }


    //Format : MERGED TCS Grouped (Uniformal)
    // READ_W_AR: set configuation for AXI-ar
    READ_W_AR.whenIsActive {

      // AXI-AR Configurations
      io.AXI.ar.addr := (io.base_addr_W + (cnt_beats_W << conf.OFFSET)).resize(conf.axiConfig.addressWidth)
      io.AXI.ar.len := conf.total_numb_beat_W - 1
      io.AXI.ar.size := conf.AXI_DATA_SIZE
      io.AXI.ar.valid := True

      when(io.AXI.ar.fire) {
        goto(LOAD_W_DATA)
      }

    }


    // LOAD_W_DATA: read S indices and put into buffer , stay until we have put all data into the buffers
    LOAD_W_DATA.whenIsActive {

      io.AXI.r.ready := True // We are ready to receive data

      when(io.AXI.r.fire) { // arrives 512 bit

        io.kernel.foreach(_.valid := True)
        cnt_beats_W.increment() // for address calculation

        when(io.AXI.r.last){
          cnt_w.clear()
        }.otherwise{
          cnt_w.increment()
        }

      }

      when(io.AXI.r.last) {
        cnt_entries.increment()
        when(cnt_entries < io.Non_zero_per_K_slice - 1 ){ // read Non_zero_per_K_slice times S indexes for S/2 columns

          goto(READ_W_AR)
        }.otherwise {
         // we have read all entries for this K_slice
          cnt_entries.clear()

          when(cnt_K < io.K){
            cnt_slice.increment()
            goto(READ_X_AR)
          }.otherwise{
            // we are done with S/2 columns for 1 Row of X
            cnt_N := cnt_N + conf.S_2
            goto(WAIT)
          }


          }

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


    // We need 14 cycles to get to this state
    // MEM_ADDR_Y :  Configure AXI-AW Channel
    MEM_ADDR_Y.whenIsActive {

      // io.AXI-AW Configurations
      io.AXI.aw.addr := (io.base_addr_Y + (cnt_beats_Y.value << conf.OFFSET)).resize(conf.axiConfig.addressWidth)
      io.AXI.aw.len := conf.total_numb_beat_Y - 1
      io.AXI.aw.size := conf.AXI_DATA_SIZE
      io.AXI.aw.valid := True

      when(io.AXI.aw.fire) {
        goto(MEM_WRITE_Y)
      }


    }
    // MEM_WRITE: put data into w.data buffer
    MEM_WRITE_Y.whenIsActive {

      // Counter to control AXI.w.last
      val cnt_beats_y = Counter(16 bits, io.AXI.w.fire)

      // TODO if we really need to reverse
      // AXI-W Configuration
      io.AXI.w.data := (io.result_GEMM.reverse).map(_.asBits).reduce(_##_).subdivideIn(conf.axiConfig.dataWidth bits)  (cnt_beats_y.resize(log2Up(conf.total_numb_beat_Y)))
      io.AXI.w.last := cnt_beats_y.value === conf.total_numb_beat_Y
      io.AXI.w.valid := True
      io.AXI.b.ready.set()


      when(io.AXI.w.fire) {
        cnt_beats_Y.increment()
      }

      when(io.AXI.b.fire) { // received feedback
       io.reset_acc := True // reset accumulator
        cnt_K := 0
        when(cnt_N < io.N) {
          cnt_slice.clear()
          goto(READ_X_AR)// if we have columns of W left
        }.otherwise{
          cnt_N := 0
          goto(NEXT_ROW)
        }
      }

    }
  }
}
