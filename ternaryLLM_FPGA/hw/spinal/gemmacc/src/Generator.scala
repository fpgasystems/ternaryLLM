package gemmacc.src

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._
import spinal.lib.bus.amba4.axi._
import gemmacc.coyote._
import gemmacc.src.design.{PE, WrapGEMM, _}
import MatrixAdd.Wrap_logic
import gemmacc.util._

object MySpinalConfig extends SpinalConfig(
  targetDirectory = "hw/gen/",
  defaultConfigForClockDomains = ClockDomainConfig(
    resetKind = SYNC,
    resetActiveLevel = LOW
  )
)


object TernaryGEMM {
  def main(args: Array[String]): Unit = {
    val sysConf = new ConfigSys {}
    MySpinalConfig.generateVerilog {
      val top = new WrapGEMM(sysConf)
      top.setDefinitionName("ternaryGEMM")
      top
    }
  }
}

object PE {
  def main(args: Array[String]): Unit = {
    val sysConf = new ConfigSys {}
    MySpinalConfig.generateVerilog {
      val top = new PE(sysConf)
      top.setDefinitionName("PE")
      top
    }
  }
}



object TestLogic {
  def main(args: Array[String]): Unit = {
    val sysConf = new ConfigSys {}
    MySpinalConfig.generateVerilog{
      val top = new Wrap_logic(sysConf)
      top.setDefinitionName("testLogic")
      top
    }
  }
}
