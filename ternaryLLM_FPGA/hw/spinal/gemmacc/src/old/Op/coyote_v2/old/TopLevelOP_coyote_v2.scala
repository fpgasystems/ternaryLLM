
// TODO: Clean code

package gemmacc.src.Op.coyote_v2.old

import gemmacc.coyote.{AXI4SR, ReqT, ack_t}
import gemmacc.src.Base.{ActivationBuffer, TernaryGEMM}
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.{master, slave}

import scala.language.postfixOps

// Connect all Module together
class TopLevelOP_coyote_v2(conf : ConfigSys) extends Component{

  val io = new Bundle {
    val start = in Bool()
    val done = out Bool()
    val M = in UInt(16 bits)
    val N = in UInt(16 bits)
    val K = in UInt(16 bits)
    val Non_zero_per_K_slice = in UInt(16 bits)
    val base_addr_X = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt(conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt(conf.axiConfig.addressWidth bits)

    val sq_rd = master Stream(ReqT())
    val sq_wr = master Stream(ReqT())
    val cq_rd = slave Stream(ack_t())
    val cq_wr = slave Stream(ack_t())

    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

  }

  // Generate Hardware at compile Time
  val dataFSM = new DataFSM_OP_coyote_v2(conf)
  val GEMM = Seq.tabulate(conf.UNROLL_M)(i => new TernaryGEMM(conf.S, conf.BIT_WIDTH_X_Y))
  val xBuffer = Seq.tabulate(conf.UNROLL_M)(i => new ActivationBuffer(conf.S, conf.K_slice, conf.BIT_WIDTH_INDEX, conf.BIT_WIDTH_X_Y))

  // Connect read and write queues and axi's
  dataFSM.io.sq_rd <> io.sq_rd
  dataFSM.io.sq_wr <> io.sq_wr
  dataFSM.io.cq_rd <> io.cq_rd
  dataFSM.io.cq_wr <> io.cq_wr
  dataFSM.io.axis_card_send <> io.axis_card_send
  dataFSM.io.axis_card_recv <> io.axis_card_recv


  // Connect Kernel
  for(j <- 0 until conf.UNROLL_M) {
    xBuffer(j).io.kernel <> dataFSM.io.kernel(j)
    GEMM(j).io.buffer <> xBuffer(j).io.output
    }

  // Connect Runtime Inputs
  dataFSM.io.M <> io.M
  dataFSM.io.N <> io.N
  dataFSM.io.K <> io.K
  dataFSM.io.base_addr_X <> io.base_addr_X
  dataFSM.io.base_addr_W <> io.base_addr_W
  dataFSM.io.base_addr_Y <> io.base_addr_Y
  dataFSM.io.Non_zero_per_K_slice <> io.Non_zero_per_K_slice



  for(j <- 0 until conf.UNROLL_M) {
  xBuffer(j).io.x <> dataFSM.io.x(j)
    GEMM(j).io.start <> dataFSM.io.reset_acc
    dataFSM.io.result_GEMM(j)<> GEMM(j).io.output
  }

  // Connect Start Signal , Finished Signal
  dataFSM.io.start <> io.start
  io.done := dataFSM.io.done
}

