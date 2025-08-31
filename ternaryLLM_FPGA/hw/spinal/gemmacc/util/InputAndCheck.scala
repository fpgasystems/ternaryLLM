

package gemmacc.util

import java.nio.{ByteBuffer, ByteOrder}
import scala.util.Random

// Helper function to create X and W values and to generate Naive Solution

object InputAndCheck {


  // Generate activation X
  def generateX(M: Int, K: Int) : Array[Byte] = {
    val buffer_X = ByteBuffer.allocate(M * K) // 2 Bytes per short
    buffer_X .order(ByteOrder.LITTLE_ENDIAN)
    val X = Array.tabulate(M,K)((i,j) => if (j % 2 == 0){5.toByte} else{1.toByte}).flatten
    X.foreach(i => buffer_X.put(i))

    return  buffer_X.array()
  }

  // Generate W_index
  def generateW(S:Int, entries: Int, N : Int) : Array[Byte]  = {
    val rand = Random
    val times = N / (S/2)
    val buffer_W = ByteBuffer.allocate(times * entries* S ) // 1 Byte per entries, if we want 2
    buffer_W.order(ByteOrder.LITTLE_ENDIAN)
    val W = Array.tabulate(times * entries,S)((i,j) => if (j % 2 == 0){0.toByte} else{1.toByte})
    W.foreach(i => buffer_W.put(i))
    return buffer_W.array()
  }


  // Naive GEMM for ternary sparse weights (W_idx) and dense X
  def naiveGEMM(M: Int, N: Int, K: Int, S: Int, entries: Int, X: Array[Byte], W: Array[Byte]): Array[Array[Int]] = {

    val S_2 = S / 2
    val Y = Array.ofDim[Int](M, N)
    val entries_per_K_slice = (entries / (K / 128))

    // Convert X (Byte[]) to 2D Short array (M x K)
    val X_buffer = ByteBuffer.wrap(X).order(ByteOrder.LITTLE_ENDIAN)
    val X_matrix = Array.ofDim[Byte](M, K)
    for (m <- 0 until M; k <- 0 until K) {
      X_matrix(m)(k) = X_buffer.get(m * K + k)
    }

    // Convert W (Byte[]) to 2D Short array (times_entries x N)
    val times = N / (S_2)
    val times_entries = times * entries
    val W_buffer = ByteBuffer.wrap(W).order(ByteOrder.LITTLE_ENDIAN)
    val W_matrix = Array.ofDim[Byte](times_entries, S)
    for (k <- 0 until times_entries ; n <- 0 until S) {
      W_matrix(k)(n) = W_buffer.get(k * S + n)
    }

    var slice_weights = 0

    for (m <- 0 until M) {
      slice_weights = 0
      for (n <- 0 until N) {

        // calculate column group and offset(inside that group)
        val col_group = n / (S_2)
        val col_offset = n % (S_2)

        for (e <- 0 until entries) {
          val base = col_group * entries + e

          if(entries % (entries_per_K_slice - 1) == 0){
            slice_weights = slice_weights + 1
          }

          // Extract pos and negative index
          val pos = W_matrix(base)(2 * col_offset).toInt
          val neg = W_matrix(base)(2 * col_offset + 1).toInt

          // Extract non-zero-values of X and put it into pos and negativ values
          val x_pos =  X_matrix(m)(pos + slice_weights * 256)
          val x_neg =  X_matrix(m)(neg + slice_weights *256)

          // Update Y entries
          Y(m)(n) += (x_pos - x_neg)
        }
      }
    }

    Y
  }





}
