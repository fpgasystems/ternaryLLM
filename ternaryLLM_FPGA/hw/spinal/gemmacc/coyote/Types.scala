//Source : https://github.com/yiweifengyan/dlm/blob/master/dlm/src/main/scala/hwsys/util/Helpers.scala


package gemmacc.coyote

import spinal.core._
import spinal.lib._

// For Coyote V2
case class ReqT() extends Bundle {
  val opcode  = UInt(5 bits)
  val strm    = UInt(2 bits)
  val mode    = Bool()
  val rdma    = Bool()
  val remote  = Bool()

  val vfid    = UInt(4 bits)
  val pid     = UInt(6 bits)
  val dest    = UInt(4 bits)

  val last    = Bool()

  val vaddr   = UInt(48 bits)
  val len     = UInt(28 bits)

  val actv    = Bool()
  val host    = Bool()
  val offs    = UInt(6 bits)
  // Compute reserved bits to fill remaining 128-bit wide struct
  // Total used bits so far = 109
  val rsrvd   = UInt(19 bits)  // 128 - 109
}

case class ack_t() extends Bundle {
  val opcode = UInt(5 bits)
  val strm   = UInt(2 bits)
  val remote = Bool()
  val host   = Bool()
  val dest   = UInt(4 bits)
  val pid    = UInt(6 bits)
  val vfid   = UInt(4 bits)
  val rsrvd  = UInt(9 bits)
}


// AXI_DATA_BITS = 512
// AXI_ID_BITS = 6
case class AXI4SR() extends Bundle with IMasterSlave {
  val tdata = Bits(512 bits)
  val tkeep = Bits(64 bits) // 512 / 8
  val tid = UInt(6 bits)
  val tlast  = Bool()
  val tvalid = Bool()
  val tready = Bool()

  override def asMaster(): Unit = {
    out(tdata, tkeep, tid, tlast, tvalid)
    in(tready)
  }

  override def asSlave(): Unit = {
    out(tready)
    in(tdata,tkeep,tlast,tvalid,tid)
  }

  def tieOffMaster(): Unit = {
    tdata  := 0
    tkeep  := 0
    tid    := 0
    tlast  := False
    tvalid := False
  }

  def tieOffSlave(): Unit = {
    tready := False
  }

}