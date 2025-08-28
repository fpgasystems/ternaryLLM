
// TODO: Clean code

package gemmacc.src.Base

import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4
import spinal.lib.master

import scala.language.postfixOps

// Connect all Module together
class TopLevelBase(conf : ConfigSys) extends Component{

  val io = new Bundle {
    val start = in Bool()
    val M = in UInt(16 bits)
    val N = in UInt(16 bits)
    val K = in UInt(16 bits)
    val base_addr_X = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt(conf.axiConfig.addressWidth bits)
    val Non_zero_per_K_slice = in UInt(16 bits)
    val AXI = master(Axi4(conf.axiConfig))
    val done = out Bool()
  }

  // Generate Hardware at compile Time
  val dataFSM = new DataFSM_Base(conf)
  val GEMM = new TernaryGEMM(conf.S ,conf.BIT_WIDTH_X_Y)
  val xBuffer = new ActivationBuffer(conf.S, conf.K_slice, conf.BIT_WIDTH_INDEX, conf.BIT_WIDTH_X_Y)

  // for simulation
  dataFSM.io.AXI <> io.AXI

  // Connect Kernel
  xBuffer.io.kernel <> dataFSM.io.kernel
  GEMM.io.buffer <> xBuffer.io.output


  // Connect Runtime Inputs
  dataFSM.io.M <> io.M
  dataFSM.io.N <> io.N
  dataFSM.io.K <> io.K
  dataFSM.io.base_addr_X <> io.base_addr_X
  dataFSM.io.base_addr_W <> io.base_addr_W
  dataFSM.io.base_addr_Y <> io.base_addr_Y
  dataFSM.io.Non_zero_per_K_slice <> io.Non_zero_per_K_slice


  // Connect Values of X
  xBuffer.io.x <> dataFSM.io.x

  // Connect Start Signal
  dataFSM.io.start <> io.start
  GEMM.io.start <> dataFSM.io.reset_acc

  // Connect ternary GEMM output
  dataFSM.io.result_GEMM <> GEMM.io.output

  io.done := dataFSM.io.done

}

