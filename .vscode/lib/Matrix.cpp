#include "Matrix.h"

// Matrix is header-only. These explicit instantiations just make sure every
// supported element type is always compiled, even if nothing uses it yet.
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;
