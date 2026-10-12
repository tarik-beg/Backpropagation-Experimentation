#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Matrix.h"

/*
    Streams MNIST samples straight from the IDX files, so only the current image is ever in memory.

    File format (header integers are 32-bit big-endian):
        images: magic 2051, count, rows, cols, then count*rows*cols pixel bytes (0-255), row by row
        labels: magic 2049, count, then count label bytes (0-9)
*/
class MNISTReader{
private:
    static constexpr std::streamoff ImagesHeaderSize=16;
    static constexpr std::streamoff LabelsHeaderSize=8;

    std::ifstream _images;
    std::ifstream _labels;

    size_t _size;
    size_t _rows;
    size_t _cols;
    size_t _next; //index of the sample Next() returns

    std::vector<uint8_t> _pixelBuffer;

    static uint32_t ReadBigEndian(std::ifstream& file){
        unsigned char bytes[4];
        if(!file.read(reinterpret_cast<char*>(bytes), 4)){
            throw std::runtime_error("MNISTReader: file ends inside the header");
        }
        return (uint32_t(bytes[0])<<24) | (uint32_t(bytes[1])<<16) | (uint32_t(bytes[2])<<8) | uint32_t(bytes[3]);
    }

    void Seek(size_t index){
        //clear() resets the end-of-file flag so seekg works after reading the last sample
        _images.clear();
        _labels.clear();
        _images.seekg(ImagesHeaderSize + static_cast<std::streamoff>(index*ImageSize()));
        _labels.seekg(LabelsHeaderSize + static_cast<std::streamoff>(index));
        _next=index;
    }

public:
    /*
        Paths to the MNIST files, relative to the project root (the working directory when run from VS Code).
        e.g. MNISTReader train(MNISTReader::TrainImagesPath, MNISTReader::TrainLabelsPath);
    */
    inline static const std::string TrainImagesPath="MNISTData/train-images.idx3-ubyte";
    inline static const std::string TrainLabelsPath="MNISTData/train-labels.idx1-ubyte";
    inline static const std::string TestImagesPath="MNISTData/t10k-images.idx3-ubyte";
    inline static const std::string TestLabelsPath="MNISTData/t10k-labels.idx1-ubyte";

    MNISTReader(const std::string& imagesPath, const std::string& labelsPath)
        : _images(imagesPath, std::ios::binary), _labels(labelsPath, std::ios::binary), _next(0){
        if(!_images){
            throw std::runtime_error("MNISTReader: can't open " + imagesPath);
        }
        if(!_labels){
            throw std::runtime_error("MNISTReader: can't open " + labelsPath);
        }
        if(ReadBigEndian(_images)!=2051){
            throw std::runtime_error("MNISTReader: " + imagesPath + " is not an IDX image file");
        }
        if(ReadBigEndian(_labels)!=2049){
            throw std::runtime_error("MNISTReader: " + labelsPath + " is not an IDX label file");
        }

        _size=ReadBigEndian(_images);
        _rows=ReadBigEndian(_images);
        _cols=ReadBigEndian(_images);
        if(ReadBigEndian(_labels)!=_size){
            throw std::runtime_error("MNISTReader: image and label files have different sample counts");
        }

        _pixelBuffer.resize(ImageSize());
    }

    size_t Size() const { return _size; }
    size_t Rows() const { return _rows; }
    size_t Cols() const { return _cols; }
    size_t ImageSize() const { return _rows*_cols; }

    /*
        Reads the next sample in file order and returns false once every sample has been read.
        image becomes an ImageSize() x 1 column vector (pixels row by row, scaled to [0,1]),
        ready to use as layer 0. label is the digit 0-9. Passing the same image every call reuses its memory.
    */
    bool Next(MatrixF& image, int& label){
        if(_next>=_size){
            return false;
        }

        char labelByte;
        if(!_images.read(reinterpret_cast<char*>(_pixelBuffer.data()), static_cast<std::streamsize>(_pixelBuffer.size()))
            || !_labels.read(&labelByte, 1)){
            throw std::runtime_error("MNISTReader: file is shorter than its header says");
        }

        if(image.rows()!=ImageSize() || image.cols()!=1){
            image=MatrixF(ImageSize(), 1);
        }
        for(size_t p=0;p<ImageSize();p++){
            image(p, 0)=_pixelBuffer[p]/255.0f;
        }
        label=static_cast<unsigned char>(labelByte);

        _next++;
        return true;
    }

    /*
        Reads sample number index (0 to Size()-1). Next() then continues from index+1.
        Use with a shuffled list of indices to go through the data in random order.
    */
    void Get(size_t index, MatrixF& image, int& label){
        if(index>=_size){
            throw std::out_of_range("MNISTReader: sample index out of range");
        }
        Seek(index);
        Next(image, label);
    }

    //Goes back to the first sample, e.g. at the start of each epoch
    void Reset(){
        Seek(0);
    }
};
