#pragma once
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Simple row-major matrix. Not optimized, just easy to use and modify.
// Pick the element type with the template argument, or use the aliases below:
//   MatrixF = float, MatrixD = double, MatrixI = int
template <typename T = double>
class Matrix {
    static_assert(std::is_arithmetic_v<T>, "Matrix element type must be arithmetic");

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;

public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols, T value = T(0))
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<T>> values)
        : rows_(values.size()), cols_(values.size() ? values.begin()->size() : 0) {
        data_.reserve(rows_ * cols_);
        for (const auto& row : values) {
            if (row.size() != cols_) {
                throw std::invalid_argument("Matrix: rows have different lengths");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // Copies are deep. Moves take the data and leave the source as an empty 0x0 matrix.
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_.clear();
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
            other.data_.clear();
        }
        return *this;
    }

    static Matrix identity(std::size_t n) {
        Matrix m(n, n);
        for (std::size_t i = 0; i < n; i++) {
            m(i, i) = T(1);
        }
        return m;
    }

    // Uniform random values in [lo, hi]
    static Matrix random(std::size_t rows, std::size_t cols, T lo = T(-1), T hi = T(1)) {
        static std::mt19937 gen(std::random_device{}());
        Matrix m(rows, cols);
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(lo, hi);
            for (T& x : m.data_) {
                x = dist(gen);
            }
        } else {
            std::uniform_real_distribution<T> dist(lo, hi);
            for (T& x : m.data_) {
                x = dist(gen);
            }
        }
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Bounds-checked element access: m(row, col)
    T& operator()(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix: index out of range");
        }
        return data_[r * cols_ + c];
    }

    T operator()(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix: index out of range");
        }
        return data_[r * cols_ + c];
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix: size mismatch in addition");
        }
        for (std::size_t i = 0; i < data_.size(); i++) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix: size mismatch in subtraction");
        }
        for (std::size_t i = 0; i < data_.size(); i++) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Matrix& operator*=(T scalar) {
        for (T& x : data_) {
            x *= scalar;
        }
        return *this;
    }

    // The && versions of +, -, *scalar, hadamard and apply run when the left operand is a temporary
    // (e.g. the result of W*a) and work on it in place. The const& versions copy once, then use the && version.
    Matrix operator+(const Matrix& other) const& { return Matrix(*this) + other; }
    Matrix operator+(const Matrix& other) && {
        *this += other;
        return std::move(*this);
    }

    Matrix operator-(const Matrix& other) const& { return Matrix(*this) - other; }
    Matrix operator-(const Matrix& other) && {
        *this -= other;
        return std::move(*this);
    }

    Matrix operator*(T scalar) const& { return Matrix(*this) * scalar; }
    Matrix operator*(T scalar) && {
        *this *= scalar;
        return std::move(*this);
    }

    // Matrix product
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix: size mismatch in multiplication");
        }
        Matrix result(rows_, other.cols_);
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < other.cols_; j++) {
                T sum = T(0);
                for (std::size_t k = 0; k < cols_; k++) {
                    sum += data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
                }
                result.data_[i * other.cols_ + j] = sum;
            }
        }
        return result;
    }

    // Element-wise product
    Matrix hadamard(const Matrix& other) const& { return Matrix(*this).hadamard(other); }
    Matrix hadamard(const Matrix& other) && {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix: size mismatch in hadamard product");
        }
        for (std::size_t i = 0; i < data_.size(); i++) {
            data_[i] *= other.data_[i];
        }
        return std::move(*this);
    }

    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                result.data_[j * rows_ + i] = data_[i * cols_ + j];
            }
        }
        return result;
    }

    // Returns a matrix with f applied to every element (e.g. sigmoid)
    Matrix apply(const std::function<T(T)>& f) const& { return Matrix(*this).apply(f); }
    Matrix apply(const std::function<T(T)>& f) && {
        for (T& x : data_) {
            x = f(x);
        }
        return std::move(*this);
    }

    void fill(T value) {
        for (T& x : data_) {
            x = value;
        }
    }

    void print(std::ostream& os = std::cout) const {
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                os << data_[i * cols_ + j] << (j + 1 < cols_ ? " " : "");
            }
            os << '\n';
        }
    }

    // Defined as non-template friends so `2 * m` converts the scalar implicitly
    friend Matrix operator*(T scalar, const Matrix& m) { return m * scalar; }
    friend Matrix operator*(T scalar, Matrix&& m) { return std::move(m) * scalar; }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        m.print(os);
        return os;
    }
};

using MatrixF = Matrix<float>;
using MatrixD = Matrix<double>;
using MatrixI = Matrix<int>;
