package gemmacc.src

import spinal.core.log2Up
import spinal.lib.bus.amba4.axi.Axi4Config

trait ConfigSys {

  // AXI4Configuration
  val axiConfig = Axi4Config(
    addressWidth = 64,
    dataWidth = 512,
    idWidth = 6,
    useStrb = true,
    useBurst = true,
    useId = true, // for simulation purpose
    useLock = false,
    useRegion = false,
    useCache = false,
    useProt = false,
    useQos = false,
    useLen = true
  )

  // COYOTE V2 Values
  val VADDR_BITS = 48
  val LOCAL_READ = 1
  val LOCAL_WRITE = 2
  val STRM_CARD = 0
  val STRM_HOST = 1

  val BIT_WIDTH_X_Y = 16
  val BIT_WIDTH_INDEX = 8
  val BIT_WIDTH_INPUT  = 16
  val DATA_WIDTH = 512
  val BIT_WIDTH_X = 8

  val DATA_SIZE_X_Y_BYTE = (BIT_WIDTH_X_Y / 8)  // One Entry of X,Y in Byte
  val DATA_SIZE_W_BYTE = (BIT_WIDTH_INDEX / 8 ) // One W_Index in Byte
  val DATA_SIZE_X = (BIT_WIDTH_X / 8)

  val BYTES_PER_BEAT = axiConfig.dataWidth / 8  // Data Size in Byte
  val AXI_DATA_SIZE = log2Up(BYTES_PER_BEAT) // AXI-SIZE
  val OFFSET = log2Up(BYTES_PER_BEAT)

  // Definition for reading X & writing Y
  val ENTRIES_PER_BEAT_X_Y  = axiConfig.dataWidth / BIT_WIDTH_X_Y
  val ENTRIES_PER_BEAT_W = axiConfig.dataWidth / BIT_WIDTH_INDEX
  val ENTRIES_PER_BEAT_X = axiConfig.dataWidth / BIT_WIDTH_X

  // Unroll Factors
  val S = 64
  var K_slice = 128
  val S_2 = S / 2
  val BITWIDTH_S = log2Up(S)
  val BYTE_X = log2Up(DATA_SIZE_X)
  val BYTE_Y = log2Up(DATA_SIZE_X_Y_BYTE)
  val UNROLL_M = 4
  val UNROLL_M_0_INDEX = UNROLL_M - 1
  val SHIFT_UNROLL = log2Up(UNROLL_M)
  val four_K = 4096
  val four_K_divided_by_Byte = four_K >> BYTE_X
  val SHIFT_4K= log2Up(four_K)

  // For X_Buffer RAM
  val TOTAL_ENTRIES_BUFFER = 16384
  val BUFFERSIZE = (UNROLL_M * TOTAL_ENTRIES_BUFFER ) / ENTRIES_PER_BEAT_X // Gives the Buffersize
  val BUFFERSIZE_PER_ROW = BUFFERSIZE / UNROLL_M
  val ADDR_WIDTH_BUFFER = log2Up(BUFFERSIZE)
  val BUFFER_ENTRIES_PER_KSLICE = K_slice / ENTRIES_PER_BEAT_X
  val TOTAL_BEATS_KSLICE = (K_slice * BIT_WIDTH_X) / DATA_WIDTH
  val TOTAL_BEATS_X = (K_slice * BIT_WIDTH_X) / DATA_WIDTH
  val TOTAL_BEATS_KSLICE_0 = TOTAL_BEATS_X - 1
  val SHIFT_BUFFERSIZE_PER_ROW = log2Up(BUFFERSIZE_PER_ROW)
  val SHIFT_BUFFER_ENTRIES_PER_KSLICE = log2Up(BUFFER_ENTRIES_PER_KSLICE)
  val EXPECTED_BEATS_4K = (four_K / BYTES_PER_BEAT) - 1

  // Data per Unroll
  val Row_Data_X = K_slice * DATA_SIZE_X_Y_BYTE // in Byte
  val Row_Data_Y = S / 2 * DATA_SIZE_X_Y_BYTE // as we write S/2 values back in memory (Byte)
  val Row_Data_W = S * DATA_SIZE_W_BYTE // in Byte
  val OFFSET_X = log2Up(UNROLL_M * DATA_SIZE_X_Y_BYTE)

  // Need to compute number of beats on runtime
  val total_numb_beat_X = if ((Row_Data_X % BYTES_PER_BEAT == 0)) Row_Data_X / BYTES_PER_BEAT else Row_Data_X / BYTES_PER_BEAT + 1 // Total Number of Beats to read 1 K_slice of X
  val total_numb_beat_Y = if ((Row_Data_Y % BYTES_PER_BEAT == 0)) Row_Data_Y / BYTES_PER_BEAT else Row_Data_Y / BYTES_PER_BEAT + 1 // Total Number of Beats for writing S/2 values of this Row
  val total_numb_beat_W = if ((Row_Data_W % BYTES_PER_BEAT == 0)) Row_Data_W / BYTES_PER_BEAT else Row_Data_W / BYTES_PER_BEAT + 1 // Total Number of Beats for reading S values of
}

class Config {

}
