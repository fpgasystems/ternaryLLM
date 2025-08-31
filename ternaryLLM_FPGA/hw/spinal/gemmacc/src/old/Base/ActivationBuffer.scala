package gemmacc.src.Base

import spinal.core._
import spinal.lib._
import spinal.lib.fsm._


class ActivationBuffer(S: Int, K_slice: Int , index_bitwidth: Int, bitwidth: Int ) extends Component{
  val io = new Bundle{
    val x = in Vec(SInt(bitwidth bits), K_slice)
    val kernel = Vec(slave Stream UInt(index_bitwidth bits), S) // S/2 kernels as we divide +1 and -1 index
    val output = Vec(master Stream SInt(bitwidth bits), S)
  }

  for(i <- 0 until S) {
    // FSM
    val FSM = new StateMachine{
      val GETINDEX = new State with EntryPoint
      val SEND_DATA = new State
      io.kernel(i).ready := isActive(GETINDEX)
      io.output(i).valid := isActive(SEND_DATA)
      io.output(i).payload := io.x(io.kernel(i).payload.resized)

      GETINDEX.whenIsActive{

        //io.kernel(i).ready := True ()
        when(io.kernel(i).fire){
         // io.output(i).data := x(io.kernel(i).data)
          goto(SEND_DATA)
        }
      }
      SEND_DATA.whenIsActive{
        //io.output(i).valid := True
        when(io.output(i).fire){
          goto(GETINDEX)
        }
      }

    }

  }


}

