// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "MemoryDeviceType.h"
#include "CUDADefines.h"

#include <stdlib.h>
#include <string.h>

#define DEVICEPTR(x) x

namespace DWIO {
    /** \brief
    Represents memory blocks, templated on the data type
    */
/*
	这个类主要就是为各种数据类型（结构体也行）分配内存（CPU or GPU），同时提供一些接口实现申请内存，释放内存，cpu和Gpu的数据交互，不同该类对象间的数据交互SetFrom（）
	想把这个实现抄过来，太方便了
*/
    template<typename T>
    class MemoryBlock {
    protected:
        bool isAllocated_CPU, isAllocated_CUDA, isMetalCompatible;
        /** Pointer to memory on CPU host. */
        DEVICEPTR(T) *data_cpu;

        /** Pointer to memory on GPU, if available. */
        DEVICEPTR(T) *data_cuda;

    public:
        enum MemoryCopyDirection {
            CPU_TO_CPU, CPU_TO_CUDA, CUDA_TO_CPU, CUDA_TO_CUDA
        };

        /** Total number of allocated entries in the data array. */
        size_t dataSize;

        /** Get the data pointer on CPU or GPU. */
        inline DEVICEPTR(T) *GetData(MemoryDeviceType memoryType)//返回这个数据类型T的数组指针
        {
            switch (memoryType) {
                case MEMORYDEVICE_CPU:
                    return data_cpu;
                case MEMORYDEVICE_CUDA:
                    return data_cuda;
            }

            return 0;
        }

        /** Get the data pointer on CPU or GPU. */
        inline const DEVICEPTR(T) *GetData(MemoryDeviceType memoryType) const {
            switch (memoryType) {
                case MEMORYDEVICE_CPU:
                    return data_cpu;
                case MEMORYDEVICE_CUDA:
                    return data_cuda;
            }

            return 0;
        }

        /** Initialize an empty memory block of the given size,
        on CPU only or GPU only or on both. CPU might also use the
        Metal compatible allocator (i.e. with 16384 alignment).
        */
        MemoryBlock(size_t dataSize, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)//初始化，给数组分配内存并赋值0（根据传入参数决定在CPU、GPU上赋值）
        {
            this->isAllocated_CPU = false;
            this->isAllocated_CUDA = false;
            this->isMetalCompatible = false;

#ifndef NDEBUG // When building in debug mode always allocate both on the CPU and the GPU
            if (allocate_CUDA) allocate_CPU = true;
#endif

            Allocate(dataSize, allocate_CPU, allocate_CUDA, metalCompatible);//根据给定参数决定是否要给cpu和Gpu分配 dataSize*sizeof(T)大小的内存
            Clear();//给分配的内存赋上默认值0
        }

        /** Initialize an empty memory block of the given size, either
        on CPU only or on GPU only. CPU will be Metal compatible if Metal
        is enabled.
        */
        MemoryBlock(size_t dataSize, MemoryDeviceType memoryType) {
            this->isAllocated_CPU = false;
            this->isAllocated_CUDA = false;
            this->isMetalCompatible = false;

            switch (memoryType) {
                case MEMORYDEVICE_CPU:
                    Allocate(dataSize, true, false, true);
                    break;
                case MEMORYDEVICE_CUDA: {
#ifndef NDEBUG // When building in debug mode always allocate both on the CPU and the GPU
                    Allocate(dataSize, true, true, true);
#else
                    Allocate(dataSize, false, true, true);
#endif
                    break;
                }
            }

            Clear();
        }

        /** Set all image data to the given @p defaultValue. */
        void Clear(unsigned char defaultValue = 0) {
            if (isAllocated_CPU) memset(data_cpu, defaultValue, dataSize * sizeof(T));
            if (isAllocated_CUDA) DWIOcudaSafeCall(cudaMemset(data_cuda, defaultValue, dataSize * sizeof(T)));

        }

        /** Resize a memory block, losing all old data.
        Essentially any previously allocated data is
        released, new memory is allocated.
        */
        void Resize(size_t newDataSize, bool forceReallocation = true) {
            if (newDataSize == dataSize) return;

            if (newDataSize > dataSize || forceReallocation) {
                bool allocate_CPU = this->isAllocated_CPU;
                bool allocate_CUDA = this->isAllocated_CUDA;
                bool metalCompatible = this->isMetalCompatible;

                this->Free();//删除之前的数据，并把相应的这些参数isAllocated_CUDA、isAllocated_CPU赋值为false
                this->Allocate(newDataSize, allocate_CPU, allocate_CUDA, metalCompatible);//重新分配内存大小
            }

            this->dataSize = newDataSize;
        }

        /** Transfer data from CPU to GPU, if possible. */
        void UpdateDeviceFromHost() const {
            if (isAllocated_CUDA && isAllocated_CPU)
                DWIOcudaSafeCall(cudaMemcpy(data_cuda, data_cpu, dataSize * sizeof(T), cudaMemcpyHostToDevice));

        }

        /** Transfer data from GPU to CPU, if possible. */
        void UpdateHostFromDevice() const {//将gpu上数据传到cpu上
            if (isAllocated_CUDA && isAllocated_CPU)
                DWIOcudaSafeCall(cudaMemcpy(data_cpu, data_cuda, dataSize * sizeof(T), cudaMemcpyDeviceToHost));
        }

        /** Copy data *///根据传入参数的不同方向将数据传给source不同设备(cpu or gpu)上的内存
        void SetFrom(const MemoryBlock<T> *source, MemoryCopyDirection memoryCopyDirection) {
            Resize(source->dataSize);
            switch (memoryCopyDirection) {
                case CPU_TO_CPU:
                    memcpy(this->data_cpu, source->data_cpu, source->dataSize * sizeof(T));
                    break;
                case CPU_TO_CUDA:
                    DWIOcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cpu, source->dataSize * sizeof(T), cudaMemcpyHostToDevice));
                    break;
                case CUDA_TO_CPU:
                    DWIOcudaSafeCall(cudaMemcpy(this->data_cpu, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToHost));
                    break;
                case CUDA_TO_CUDA:
                    DWIOcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
                    break;
                default:
                    break;
            }
        }

        /** Get an individual element of the memory block from either the CPU or GPU. */
        //将指定位置的数据返回（cpu or gpu）
        T GetElement(int n, MemoryDeviceType memoryType) const {
            switch (memoryType) {
                case MEMORYDEVICE_CPU: {
                    return this->data_cpu[n];
                }
                case MEMORYDEVICE_CUDA: {
                    T result;
                    DWIOcudaSafeCall(cudaMemcpy(&result, this->data_cuda + n, sizeof(T), cudaMemcpyDeviceToHost));
                    return result;
                }
                default:
                    throw std::runtime_error("Invalid memory type");
            }
        }

        virtual ~MemoryBlock() { this->Free(); }

        /** Allocate image data of the specified size. If the
        data has been allocated before, the data is freed.
        */
        void Allocate(size_t dataSize, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible) {
            Free();//先释放原来的内存，再申请空间，并把参数isAllocated_CPU、isAllocated_GPU复制成对应的传入参数

            this->dataSize = dataSize;

            if (allocate_CPU) {
                int allocType = 0;
                if (allocate_CUDA) allocType = 1;
                switch (allocType) {
                    case 0:
                        if (dataSize == 0) data_cpu = NULL;
                        else data_cpu = new T[dataSize];
                        break;
                    case 1:
                        if (dataSize == 0) data_cpu = NULL;
                        else
                            DWIOcudaSafeCall(cudaMallocHost((void **) &data_cpu, dataSize * sizeof(T)));//最外层的判断cuda函数有没有安全调用
                        break;
                }

                this->isAllocated_CPU = allocate_CPU;
                this->isMetalCompatible = metalCompatible;
            }

            if (allocate_CUDA) {
                if (dataSize == 0) data_cuda = NULL;
                else
                    DWIOcudaSafeCall(cudaMalloc((void **) &data_cuda, dataSize * sizeof(T)));
                this->isAllocated_CUDA = allocate_CUDA;
            }
        }

        void Free()//释放空间并把对应的isAllocated_CPU、isAllocated_CUDA赋值为false
        {
            if (isAllocated_CPU) {
                int allocType = 0;
                if (isAllocated_CUDA) allocType = 1;
                switch (allocType) {
                    case 0:
                        if (data_cpu != NULL) delete[] data_cpu;
                        break;
                    case 1:
                        if (data_cpu != NULL) DWIOcudaSafeCall(cudaFreeHost(data_cpu));
                        break;
                }

                isMetalCompatible = false;
                isAllocated_CPU = false;
            }

            if (isAllocated_CUDA) {
                if (data_cuda != NULL) DWIOcudaSafeCall(cudaFree(data_cuda));
                isAllocated_CUDA = false;
            }
        }

        void Swap(MemoryBlock<T> &rhs)//这里该不会有问题把
        {
            std::swap(this->dataSize, rhs.dataSize);
            std::swap(this->data_cpu, rhs.data_cpu);
            std::swap(this->data_cuda, rhs.data_cuda);
            std::swap(this->isAllocated_CPU, rhs.isAllocated_CPU);
            std::swap(this->isAllocated_CUDA, rhs.isAllocated_CUDA);
            std::swap(this->isMetalCompatible, rhs.isMetalCompatible);
        }

        // Suppress the default copy constructor and assignment operator禁用默认的拷贝构造函数和赋值符 ’=’
        MemoryBlock(const MemoryBlock &);

        MemoryBlock &operator=(const MemoryBlock &);
    };
}


