package MatrixAdd

import gemmacc.coyote._
import gemmacc.src.ConfigSys
import spinal.core._
import spinal.core.fiber.Handle.initImplicit
import spinal.lib.{Stream, master, slave}
import spinal.lib.fsm._

import scala.language.postfixOps

// Simple Approach to test Coyote v2 Behaviour
// Read a X,W both Vector of Size 10 , add them together and store it onto Y

class logic(conf: ConfigSys) extends Component {

  val io = new Bundle{
    val start = in Bool()
    val done = out Bool()
    val base_addr_X = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_W = in UInt (conf.axiConfig.addressWidth bits)
    val base_addr_Y = in UInt (conf.axiConfig.addressWidth bits)

    val sq_rd = master Stream(ReqT())
    val sq_wr = master Stream(ReqT())
    val cq_rd = slave Stream(ack_t())
    val cq_wr = slave Stream(ack_t())

    val axis_card_recv = slave(AXI4SR())
    val axis_card_send = master(AXI4SR())

  }

  io.done := False

  val vecX = Reg(Vec(UInt(16 bits), 10))
  val vecW = Reg(Vec(UInt(16 bits), 10))
  val vecY = Reg(Vec(UInt(16 bits), 10))

  // Default assignments

  // sq_rd
  io.sq_rd.valid := False
  io.sq_rd.payload := ReqT().getZero

  // sq_wr
  io.sq_wr.valid := False
  io.sq_wr.payload := ReqT().getZero

  // cq_rd
  io.cq_rd.ready := False
  // cq_wr
  io.cq_wr.ready := False

  // receive axi
  io.axis_card_recv.tready := False

  // send axi
  io.axis_card_send.tdata := 0
  io.axis_card_send.tkeep := 0
  io.axis_card_send.tid := 0
  io.axis_card_send.tlast := False
  io.axis_card_send.tvalid := False

  val FSM = new StateMachine {
    val IDLE = new State with EntryPoint
    val READ_X,Wait_X,READ_W, WAIT_W, COMPUTE,WRITE_Y, Wait_Y = new State


    IDLE.whenIsActive{

      when(io.start && !io.done){
        goto(READ_X)
      }
    }

    READ_X.whenIsActive {

      // Set Read configurations same as AR before

      io.sq_rd.valid := True
      io.sq_rd.payload.vaddr := io.base_addr_X.resized
      io.sq_rd.payload.len := 64 // 20 Bytes :  1 Beat // TODO: code be that here is meant how long the data is
      io.sq_rd.payload.strm := conf.STRM_CARD // STRM_CARD = 0
      io.sq_rd.payload.opcode := conf.LOCAL_READ // LOCAL_READ  = 0
      io.sq_rd.payload.dest := 0 // index of axis streams :axis_card_recv[0]
      io.sq_rd.payload.last := True

      when(io.sq_rd.fire) {
        goto(Wait_X)
      }
    }

    Wait_X.whenIsActive{
      // complete Queue is ready and we are ready to receive data on the axis_card
      io.cq_rd.ready := True
      io.axis_card_recv.tready := True

      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready){ // axi is fire
        for (i <- 0 until 10) {
          vecX(i) := io.axis_card_recv.tdata(16 * (i + 1) - 1 downto 16 * i).asUInt
        }
        goto(READ_W)
      }

    }

    READ_W.whenIsActive{
      io.sq_rd.valid := True
      io.sq_rd.payload.vaddr := io.base_addr_W.resized
      io.sq_rd.payload.len := 64
      io.sq_rd.payload.strm := conf.STRM_CARD
      io.sq_rd.payload.opcode := conf.LOCAL_READ
      io.sq_rd.payload.dest := 0
      io.sq_rd.payload.last := True

      when(io.sq_rd.fire){
        goto(WAIT_W)
      }
    }

    WAIT_W.whenIsActive{
      io.cq_rd.ready := True
      io.axis_card_recv.tready := True
      when(io.axis_card_recv.tvalid && io.axis_card_recv.tready){
        for(i <- 0 until 10){
          vecW(i) := io.axis_card_recv.tdata(16 * (i + 1) - 1 downto 16 * i).asUInt
        }
        goto(COMPUTE)
      }
    }


    COMPUTE.whenIsActive{
      // Calculate Y = X + W
      for(i <- 0 until 10){
        vecY(i) := vecX(i) + vecW(i)
      }
      goto(WRITE_Y)

    }

    WRITE_Y.whenIsActive{
      io.sq_wr.valid := True
      io.sq_wr.payload.vaddr := io.base_addr_Y.resized
      io.sq_wr.payload.len := 64// 20 Bytes : 10 x 2 B : 64 corresponds to 1 Beat
      io.sq_wr.payload.strm := conf.STRM_CARD
      io.sq_wr.payload.opcode := conf.LOCAL_WRITE
      io.sq_wr.payload.dest := 0
      io.sq_wr.payload.last := True

      when(io.sq_wr.fire){
        goto(Wait_Y)
      }

    }

    Wait_Y.whenIsActive{

      io.cq_wr.ready := True

      io.axis_card_send.tvalid := True
      io.axis_card_send.tdata := (vecY.reverse).map(_.asBits).reduce(_##_).resize(conf.axiConfig.dataWidth)
      io.axis_card_send.tkeep := B(0xFFFFF, 64 bits)
      io.axis_card_send.tlast := True // as we only need to read once

      when(io.cq_wr.fire){
        io.done := True
        goto(IDLE)
      }


    }

  }

}
