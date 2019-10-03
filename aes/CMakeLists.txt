include_directories(.)
include_directories(tiny-AES-c)

set(HEADER_FILES
    tiny-AES-c/aes.h
    tiny-AES-c/aes.hpp
    common.h
    cpu.h
    gpu.h
    )

set(SOURCE_FILES
    common.cu
    cpu.cu
    gpu.cu
    tiny-AES-c/aes.c
    )

list(SORT HEADER_FILES)
list(SORT SOURCE_FILES)

source_group(Headers FILES ${HEADER_FILES})
source_group(Sources FILES ${SOURCE_FILES})

cuda_add_library(aes
    ${SOURCE_FILES}
    ${HEADER_FILES}
    OPTIONS -arch=sm_75
    )
