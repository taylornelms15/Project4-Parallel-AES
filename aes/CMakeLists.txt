set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
    "naive.h"
    "naive.cu"
    )

cuda_add_library(aes
    ${SOURCE_FILES}
    OPTIONS -arch=sm_75
    )
