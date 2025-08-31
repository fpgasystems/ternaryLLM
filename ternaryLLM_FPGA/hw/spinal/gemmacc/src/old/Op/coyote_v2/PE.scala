package gemmacc.src.Op.coyote_v2

import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

import scala.language.postfixOps

class PE(conf: ConfigSys) extends Component{

  val io = new Bundle {
    // Inputs
    val x = in Bits (conf.DATA_WIDTH bits) // Input 512-Bit X-values from RAM driven by DataFSM
    val w = slave Stream Bits (conf.DATA_WIDTH bits) // Input Stream 512-Bit W-values driven by DataFSM
    val enable_X = in Bits (8 bits) // ONE hot , to update x , driven by DataFSM
    val reset_acc = in Bool() // to reset the accumulators, driven by DatFSM
    // Output
    val y = out Bits (conf.DATA_WIDTH bits) // Output 512 Bit Y-values
    val done_acc = out Bool()
  }


  val x_reg = Vec(Reg(SInt(conf.BIT_WIDTH_X_Y bits)).init(0),conf.K_slice)  // x_reg = 256(KSCLICE) * 16 bit = 8 * 512bit
  val acc = Vec(Reg(SInt(conf.BIT_WIDTH_X_Y bits)).init(0),conf.S) // accumulators S * 16bit , initially set to 0

  io.w.ready := False // We only need to wait for valid signal
  io.done_acc := False

  when(io.reset_acc){ // reset accumulators
    acc.clearAll()
  }

  // Write the 512-Bits from Input to x_reg
  for(i <- 0 until conf.TOTAL_BEATS_KSLICE) {  // TOTAL_BEATS_KSLICE = 8
    when(io.enable_X(i)) { // check if we need to adjust x_reg
      for(j <- 0 until conf.ENTRIES_PER_BEAT_X_Y){   // ENTRIES_PER_BEAT_X_Y = 32
        x_reg(i * conf.ENTRIES_PER_BEAT_X_Y + j) := io.x.subdivideIn(conf.ENTRIES_PER_BEAT_X_Y slices)(j).asSInt  // subdivide into conf.ENTRIES_PER_BEAT_X_Y slices gives us a Vec() of 32 * 16 bits
      }
    }
  }

  // Counter fo indexing W
  val cnt_w = Counter(conf.S)

  // Only works with 8 bit Weigths!!!!
  // FiniteState Machine to get index of X and put it into right acc
  val FSM_W = new StateMachine {
    val WAIT_WEIGHT = new State with EntryPoint
    val PROCESSING = new State

    WAIT_WEIGHT.onEntry(cnt_w.clear())
    // Clear Cnt, Wait until io.w is fire
    WAIT_WEIGHT.whenIsActive {
      io.w.ready := True
      //cnt_w.clear()
      when(io.w.fire) {
        goto(PROCESSING)
      }
    }
    // process indexing sequentially
    PROCESSING.whenIsActive{

      cnt_w.increment()
      acc(cnt_w.value) := acc(cnt_w.value) + x_reg((io.w.payload.subdivideIn(conf.ENTRIES_PER_BEAT_W slices)(cnt_w).asUInt)) // we subdivide w into chunks of 8 bit times ENTRIES_PER_BEAT(64), so we have Vec of 64=S * 8bits

      when(cnt_w.willOverflow){
       io.done_acc := True
        // MAYBE WAIT???
        goto(WAIT_WEIGHT)
      }
    }



  }

//  // X-BAR Parallel version:
//  when(io.w.fire){ // when weight is valid and ready
//    for(i <- 0 until conf.S){
//      // fetch index from k_slice and accumulate
//      acc(i) := acc(i) + x_reg(io.w.payload.subdivideIn(conf.ENTRIES_PER_BEAT_W slices)(i).asUInt) // we subdivide w into chunks of 8 bit times ENTRIES_PER_BEAT(64), so we have Vec of 64=S * 8bits
//    }
//  }

  for(i <- 0 until conf.S){
    // Substract to get output every two accumulators correspond to one output entry at the end
    if (i % 2 == 1) {
      io.y.subdivideIn(conf.S_2 slices)(i/2) := (acc(i - 1) - acc(i)).asBits
    }

  }


}
