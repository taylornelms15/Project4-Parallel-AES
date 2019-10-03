#pragma once

#include "common.h"

namespace AES {
    namespace Naive {
        AES::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
    }
}
