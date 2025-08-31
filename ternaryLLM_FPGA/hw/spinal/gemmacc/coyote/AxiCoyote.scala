package gemmacc.coyote

import gemmacc.src.ConfigSys
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4
import spinal.lib.{master, slave}

import scala.language.postfixOps

class AxiCoyote(conf: ConfigSys) extends Component {
  val io = new Bundle {

    val mem = master(Axi4(conf.axiConfig))

    val sq_rd = slave Stream(ReqT())    // Read request stream
    val sq_wr = slave Stream(ReqT())    // Write request stream
    val cq_rd = master Stream(ack_t())  // Read completion ack
    val cq_wr = master Stream(ack_t())  // Write completion ack

    val axis_card_recv = master(AXI4SR())  // Write data stream (from logic)
    val axis_card_send = slave(AXI4SR())   // Read data stream (to logic)
  }

  io.cq_rd.valid := True
  io.cq_rd.payload := ack_t().getZero

  // AR-Channel
  io.mem.ar.addr := io.sq_rd.payload.vaddr.resized
  io.mem.ar.id := io.sq_rd.payload.pid.resized
  io.mem.ar.len := Mux(io.sq_rd.payload.len % conf.BYTES_PER_BEAT === 0, io.sq_rd.payload.len / conf.BYTES_PER_BEAT - 1, io.sq_rd.payload.len / conf.BYTES_PER_BEAT).resized
  io.mem.ar.size := conf.AXI_DATA_SIZE
  io.mem.ar.setBurstINCR()
  io.mem.ar.valid :=  io.sq_rd.valid
  io.sq_rd.ready := io.mem.ar.ready

  // R Channel
  io.mem.r.ready := io.axis_card_recv.tready
  io.axis_card_recv.tvalid := io.mem.r.valid
  io.axis_card_recv.tdata := io.mem.r.data
  io.axis_card_recv.tid := 0
  io.axis_card_recv.tlast := io.mem.r.last
  io.axis_card_recv.tkeep.setAll()


  // AW-Channel
  io.mem.aw.addr := io.sq_wr.payload.vaddr.resized
  io.mem.aw.id := io.sq_wr.payload.pid.resized
  io.mem.aw.len := Mux(io.sq_wr.payload.len % conf.BYTES_PER_BEAT === 0, io.sq_wr.payload.len / conf.BYTES_PER_BEAT - 1, io.sq_wr.payload.len / conf.BYTES_PER_BEAT).resized
  io.mem.aw.size := conf.AXI_DATA_SIZE
  io.mem.aw.setBurstINCR()
  io.mem.aw.valid :=  io.sq_wr.valid
  io.sq_wr.ready := io.mem.aw.ready

  // W-Channel
  io.mem.w.data := io.axis_card_send.tdata
  io.mem.w.last := io.axis_card_send.tlast
  io.mem.w.valid := io.axis_card_send.tvalid
  io.mem.w.strb := io.axis_card_send.tkeep
  io.axis_card_send.tready := io.mem.w.ready

  // B-Channel
  io.cq_wr.valid := io.mem.b.valid
  io.mem.b.ready := io.cq_wr.ready
  io.cq_wr.payload := ack_t().getZero
}
