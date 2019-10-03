#pragma once

#include "common.h"

namespace AES {
    namespace CPU {
	    AES::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
