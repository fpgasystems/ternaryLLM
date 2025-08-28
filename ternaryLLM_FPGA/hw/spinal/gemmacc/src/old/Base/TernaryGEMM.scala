package gemmacc.src.Base

import spinal.core._
import spinal.lib._


class TernaryGEMM(S: Int, bitwidth: Int) extends Component {

  val io = new Bundle {
    // Input: Buffer from ActivationBuffer
    val start = in Bool()
    //val out_valid =  Vec(Bool(),S) // all data has been fetched, FIFOs are empty
    val buffer = Vec(slave Stream SInt(bitwidth bits), S)

    // Output: X * W
    val output = out Vec(SInt(bitwidth bits), S / 2)
  }



  // Accumulators (if start == true it resets the acc to 0)
  val acc = Vec(Reg(SInt(bitwidth + 5 bits)) init (0), S)


  for (i <- 0 until S) {
    io.buffer(i).ready := !io.start

    // Reset Accumulator
    when(io.start){
      acc.clearAll()
    }

    // once data is valid and ready accumulate
    when(io.buffer(i).fire) {
      // Accumulate
      acc(i) := acc(i) + io.buffer(i).payload
      //accDebug(i) := acc(i) // Debugging
    }

    // Substract to get output every two accumulators correspond to one output entry at the end
    if (i % 2 == 1) {
      // one cycle to update accumulator and another cycle to get the correct subtraction result
      // after read out subtraction result we can reset the out_valid == false and can start again
      io.output(i/2) := (acc(i - 1) - acc(i)).resize(bitwidth)
    }

  }

}





