//
// Created by yidong on 8/31/20.
//

#pragma once

#include <string>
#include <cstring>

enum op_t {
    eSUM = 0,
    eMAX,
    eMIN,
    eSUB,
    eMUL,
    eDIV,
};

//1D tensor
template <class T>
struct array1d_t {
    T* data_ptr;
    int64_t col_count;
    bool alloc;

    T& operator[] (int64_t index) {//returns the element 
        return data_ptr[index];
    }
    void assign (int64_t index, const T& value) {//returns the element 
        data_ptr[index] = value ;
    }
    array1d_t(int64_t a_col_count) {
        data_ptr = (T*)calloc(sizeof(T), a_col_count);
        col_count = a_col_count;
        alloc = true;
    }
    array1d_t(T* ptr, int64_t a_col_count) {
        data_ptr = ptr;
        col_count = a_col_count;
        alloc = false;
    }

    ~array1d_t() {
        if (alloc) {
            free(data_ptr);
        }
    }
    void reset() {
        memset(data_ptr, 0, col_count*sizeof(T));
    }
    void add(T* ptr) {
        T* row_ptr = data_ptr;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] += ptr[i];
        }
    }
    
    void addw(T* ptr, T weight) {
        T* row_ptr = data_ptr;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] += ptr[i]*weight;
        }
    }
};



//2D tensor
template <class T>
struct array2d_t {
    T* data_ptr;
    int64_t row_count;
    int64_t col_count;
    T* operator[] (int64_t index) {//returns a row
        return data_ptr + col_count*index;
    }
    array2d_t(T* a_ptr, int64_t a_row_count, int64_t a_col_count) {
        data_ptr = a_ptr;
        row_count = a_row_count;
        col_count = a_col_count;
    }
    void row_copy(T* ptr, int64_t index) {
        T* row_ptr = data_ptr + col_count*index;
        memcpy(row_ptr, ptr, col_count*sizeof(T)); 
    }
    void row_copy_norm(T* ptr, int64_t index, int degree) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] = ptr[i]/degree;
        }
    }
    void row_add(T* ptr, int64_t index) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] += ptr[i];
        }
    }
    void row_normalize(int64_t index, T degree) {
        T* row_ptr = data_ptr + col_count*index;
        for (int64_t i = 0; i < col_count; ++i) {
            row_ptr[i] /= degree;
        }
    }
    T get_item(int64_t row_id, int64_t col_id) {
        return data_ptr[row_id*col_count + col_id];
    }
    array1d_t<T> get_row(int64_t row_id) {
        return array1d_t<T>(data_ptr + row_id*col_count, col_count);
    }
    void reset() {
        memset(data_ptr, 0, row_count*col_count*sizeof(T));
    }
};

//3D tensor
template <class T>
struct array3d_t {
    T* data_ptr;
    int64_t matrix_count;
    int64_t row_count;
    int64_t col_count;
    T* operator[] (int64_t index) {//returns a matrix
        return data_ptr + col_count * row_count * index;
    }
    array3d_t(T* a_ptr, int64_t a_matrix_count, int64_t a_row_count, int64_t a_col_count) {
        data_ptr = a_ptr;
        matrix_count = a_matrix_count;
        row_count = a_row_count;
        col_count = a_col_count;
    }

    void matrix_copy(T* ptr, int64_t index) {
            T* row_ptr = data_ptr + row_count * col_count * index;
            memcpy(row_ptr, ptr, row_count * col_count * sizeof(T)); 
    }
    array1d_t<T> get_row(int64_t index, int64_t row_id) {
         T* row_ptr = data_ptr + row_count * col_count * index + row_id*col_count;
         array1d_t<T> array(row_ptr, col_count);
         return array;
    }
    T* get_row_ptr(int64_t index, int64_t row_id) {
         T* row_ptr = data_ptr + row_count * col_count * index + row_id*col_count;
         return row_ptr;
    }
    void row_copy(T* ptr, int64_t index, int64_t row_id) {
         T* row_ptr = data_ptr + row_count * col_count * index + row_id*col_count;
         memcpy(row_ptr, ptr, row_count * col_count * sizeof(T)); 
    }

};
