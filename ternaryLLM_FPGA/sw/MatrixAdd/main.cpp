
    // 1. Allocate Coyote Thread
    // 2. Allocate through getMem() 3 Locations X,W,Y
    // 3. Put values into X,W,Y
    // 4. Create sg_entry.sync for LOCAL_OFFLOAD and wait until it is done!
    // 5. set Registers setCSR() (only start not) and wait until it is completed
    // 6. set start signal
    // 7. wait until done_signal using getCSR
    // 8. Read out Y 
    // 9  free Memory

    #include <any>

    // Coyote-specific includes
    #include "cThread.hpp"
    //using namespace fpga;  // for master/branch
    using namespace coyote;



    // Default vFPGA to assign cThreads to
    #define DEFAULT_VFPGA_ID 0

    // Registers for axi_ctrl
    #define START 0x00
    #define DONE 0x08
    #define ADDR_X 0x10
    #define ADDR_W 0x18
    #define ADDR_Y 0x20



    int main(int argc, char *argv[]){

    // Size for allocating Memory
    u_int size = (uint) 10 * (uint) sizeof(uint16_t);

    // Create a Coyote thread and allocate memory for the vectors
    std::unique_ptr<cThread<std::any>> coyote_thread(new cThread<std::any>(DEFAULT_VFPGA_ID, getpid(), 0));
    uint16_t *X = (uint16_t *) coyote_thread->getMem({CoyoteAlloc::HPF, size});
    uint16_t *W = (uint16_t *) coyote_thread->getMem({CoyoteAlloc::HPF, size});
    uint16_t *Y = (uint16_t *) coyote_thread->getMem({CoyoteAlloc::HPF, size});
    if(!X || !W || !Y){ throw std::runtime_error("Could not allocate memory; exiting...");}

    // Initialise the input vectors and zero-out output
    for (size_t i = 0; i < 9; i++)
    {
        X[i] = 1;
        W[i] = i;
        Y[i] = 0;
    }

    // Create sg_entries
    sgEntry sg_offload_X;
    sgEntry sg_offload_W;
    sgEntry sg_Y;


    // LOCAL_OFFLOAD
   sg_offload_X.sync.addr = X ;
   sg_offload_X.sync.size = size;

   sg_offload_W.sync.addr = W;
   sg_offload_W.sync.size = size;

   sg_Y.sync.addr = Y;
   sg_Y.sync.size = size;




     std::cout << "Starting Offload W ,X, Y "<<std::endl;
    coyote_thread->invoke(CoyoteOper::LOCAL_READ,  &sg_offload_X);
    coyote_thread->invoke(CoyoteOper::LOCAL_READ,  &sg_offload_W);
    coyote_thread->invoke(CoyoteOper::LOCAL_READ,  &sg_Y);
    


    // Set registers
    std::cout << "Setting CSR: ADDR_X = 0x" << std:: hex << ( ADDR_X * 8) << ", Value = " << X << std::endl;
    coyote_thread->setCSR((uint64_t)X,ADDR_X);
    std::cout << "Get ADDR_X = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_X)) <<  std::endl;

    std::cout << "Setting CSR: ADDR_W = 0x" << std:: hex << ( ADDR_W * 8) << ", Value = " << W << std::endl;
    coyote_thread->setCSR((uint64_t)W,ADDR_W);
    std::cout << "Get ADDR_W = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_W)) <<  std::endl;

    std::cout << "Setting CSR: ADDR_Y = 0x" << std:: hex << ( ADDR_Y * 8) << ", Value = " << Y << std::endl;
    coyote_thread->setCSR((uint64_t)Y,ADDR_Y);
    std::cout << "Get ADDR_W = " << coyote_thread->getCSR(static_cast<uint32_t>(ADDR_W)) <<  std::endl;

    std::cout << "Setting CSR: START = 0x" << std:: hex << ( START * 8) << ", Value = " << 1 << std::endl;
    coyote_thread->setCSR(1,START);
    std::cout << "Get START = " << coyote_thread->getCSR(static_cast<uint32_t>(START)) <<  std::endl;



    // wait until Done
    while (coyote_thread->getCSR(DONE) != 1) {};

    std::cout << "Done Signal is high" << std::endl;

    // LOCAL_SYNC
    sg_Y.sync.addr = Y;
    sg_Y.sync.size = size; 


    coyote_thread->invoke(CoyoteOper::LOCAL_SYNC,  &sg_Y);

    // wait until done
    while(coyote_thread->checkCompleted(CoyoteOper::LOCAL_SYNC) != 1 ){} ;

    // Read out Y-Vector
    for (size_t i = 0; i < 9; i++)
    {
        std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
    }

    // Free Memory
    coyote_thread->freeMem(X);
    coyote_thread->freeMem(W);
    coyote_thread->freeMem(Y);

    return EXIT_SUCCESS;
    }
