#pragma once

#include "ITMVoxelBlockHash.h"
#include "../cuda/include/common.h"
#include <stdlib.h>

#ifndef BYTE
typedef unsigned char BYTE;
#endif



namespace DWIO
{
	class ITMMesh
	{
	public:

		struct Triangle 
		{ 	
			Vertex p0;
			Vertex p1;
			Vertex p2; 
		};

		MemoryDeviceType memoryType;

		uint noTotalTriangles;
		//static const uint noMaxTriangles_default = (SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) * 12;//* 32 * 16
#ifdef SUBMAP
		static const uint noMaxTriangles_default = (SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) * 32 * 16;//* 32 * 16
#else
		static const uint noMaxTriangles_default = (SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) * 12;
#endif
		uint noMaxTriangles;

		DWIO::MemoryBlock<Triangle> *triangles;//应该是存放顶点的
		DWIO::MemoryBlock<Triangle> *cpu_triangles; 

		explicit ITMMesh(MemoryDeviceType memoryType, uint maxTriangles = noMaxTriangles_default)
		{
			this->memoryType = memoryType;
			this->noTotalTriangles = 0;
			this->noMaxTriangles = maxTriangles;
			std::cout<<"分配内存："<<noMaxTriangles*sizeof(Triangle)/1024/1024<<"m!"<<std::endl;
			triangles = new DWIO::MemoryBlock<Triangle>(noMaxTriangles, memoryType);//总共只能有这么多顶点2^17*512*3
			std::cout<<"allocate done!"<<std::endl;
		}

		void WriteSTL(const char *fileName)
		{
			//DWIO::MemoryBlock<Triangle> *cpu_triangles; 
			bool shoulDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new DWIO::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, DWIO::MemoryBlock<Triangle>::CUDA_TO_CPU);//将顶点信息从gpu转到cpu上
				shoulDelete = true;
			}
			else cpu_triangles = triangles;

			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);

			FILE *f = fopen(fileName, "wb+");//应该是追加模式写入

			if (f != NULL) {
				for (int i = 0; i < 80; i++) fwrite(" ", sizeof(char), 1, f);

				fwrite(&noTotalTriangles, sizeof(int), 1, f);

				float zero = 0.0f; short attribute = 0;
				for (uint i = 0; i < noTotalTriangles; i++)
				{
					fwrite(&zero, sizeof(float), 1, f); fwrite(&zero, sizeof(float), 1, f); fwrite(&zero, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p2.p.x, sizeof(float), 1, f); 
					fwrite(&triangleArray[i].p2.p.y, sizeof(float), 1, f); 
					fwrite(&triangleArray[i].p2.p.z, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p1.p.x, sizeof(float), 1, f); 
					fwrite(&triangleArray[i].p1.p.y, sizeof(float), 1, f); 
					fwrite(&triangleArray[i].p1.p.z, sizeof(float), 1, f);

					fwrite(&triangleArray[i].p0.p.x, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p0.p.y, sizeof(float), 1, f);
					fwrite(&triangleArray[i].p0.p.z, sizeof(float), 1, f);

					fwrite(&attribute, sizeof(short), 1, f);

					//fprintf(f, "v %f %f %f\n", triangleArray[i].p0.x(), triangleArray[i].p0.y(), triangleArray[i].p0.z());
					//fprintf(f, "v %f %f %f\n", triangleArray[i].p1.x(), triangleArray[i].p1.y(), triangleArray[i].p1.z());
					//fprintf(f, "v %f %f %f\n", triangleArray[i].p2.x(), triangleArray[i].p2.y(), triangleArray[i].p2.z());
				}

				//for (uint i = 0; i<noTotalTriangles; i++) fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
				fclose(f);
			}

			if (shoulDelete) delete cpu_triangles;
		}

		void savePly(const std::string &filename)
		{
			std::ofstream file_out{filename};
			if (!file_out.is_open())
				return;
			DWIO::MemoryBlock<Triangle> *cpu_triangles; bool shoulDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new DWIO::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, DWIO::MemoryBlock<Triangle>::CUDA_TO_CPU);//将顶点信息从gpu转到cpu上
				shoulDelete = true;
			}
			else cpu_triangles = triangles;
			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);
			file_out << "ply" << std::endl;
			file_out << "format ascii 1.0" << std::endl;
			file_out << "element vertex " << 3*noTotalTriangles << std::endl;
			file_out << "property float x" << std::endl;
			file_out << "property float y" << std::endl;
			file_out << "property float z" << std::endl;
			// file_out << "property float nx" << std::endl;
			// file_out << "property float ny" << std::endl;
			// file_out << "property float nz" << std::endl;
			file_out << "property uchar red" << std::endl;
			file_out << "property uchar green" << std::endl;
			file_out << "property uchar blue" << std::endl;
			file_out << "end_header" << std::endl;			
			for (int i = 0; i < noTotalTriangles; ++i) {

				Vertex vertex = triangleArray[i].p2;
				Vertex vertex2 = triangleArray[i].p1;
				Vertex vertex3 = triangleArray[i].p0;
				//static_cast<int>(vertex2.c.x*255.0f)需要改成static_cast<int>(vertex2.c.x)因为在体颜色时不在除以255了
				file_out << vertex.p.x << " " << vertex.p.y << " " << vertex.p.z<<" ";
				file_out << static_cast<int>(vertex.c.x*255.0f) << " " << static_cast<int>(vertex.c.y*255.0f) << " " << static_cast<int>(vertex.c.z*255.0f) << std::endl;
				file_out << vertex2.p.x << " " << vertex2.p.y << " " << vertex2.p.z<<" ";
				file_out << static_cast<int>(vertex2.c.x*255.0f) << " " << static_cast<int>(vertex2.c.y*255.0f) << " " << static_cast<int>(vertex2.c.z*255.0f) << std::endl;
				file_out << vertex3.p.x << " " << vertex3.p.y << " " << vertex3.p.z<<" ";
				file_out << static_cast<int>(vertex3.c.x*255.0f) << " " << static_cast<int>(vertex3.c.y*255.0f) << " " << static_cast<int>(vertex3.c.z*255.0f) << std::endl;

				// file_out << normal.x << " " << normal.y << " " << normal.z << " ";
				// file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " " << static_cast<int>(color.z) << std::endl;
        	}

		}
		void saveBinPly(const std::string &filename)
		{
			std::ofstream file(filename ,std::ios::binary);
			if (!file.is_open())
				return;
			//DWIO::MemoryBlock<Triangle> *cpu_triangles; 
			bool shoulDelete = false;
			if (memoryType == MEMORYDEVICE_CUDA)
			{
				cpu_triangles = new DWIO::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
				cpu_triangles->SetFrom(triangles, DWIO::MemoryBlock<Triangle>::CUDA_TO_CPU);//将顶点信息从gpu转到cpu上
				shoulDelete = true;
			}
			else cpu_triangles = triangles;
			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);		

			std::vector<std::vector<unsigned int>>	m_FaceIndicesVertices;
			m_FaceIndicesVertices.resize(noTotalTriangles);
			for ( unsigned int i = 0; i < ( unsigned int ) noTotalTriangles; i++ )
			{
				m_FaceIndicesVertices[i].push_back(3*i+0);
				m_FaceIndicesVertices[i].push_back(3*i+1);
				m_FaceIndicesVertices[i].push_back(3*i+2);
			}

			file << "ply\n";
			file << "format binary_little_endian 1.0\n";
			file << "comment MLIB generated\n";
			file << "element vertex " << 3*noTotalTriangles << "\n";
			file << "property float x\n";
			file << "property float y\n";
			file << "property float z\n";
			file << "property uchar red\n";
			file << "property uchar green\n";
			file << "property uchar blue\n";
			file << "property uchar alpha\n";
			
			file << "element face " << noTotalTriangles<< "\n";
			file << "property list uchar int vertex_indices\n";
			file << "end_header\n";		

			float* pose =new float[3];
			unsigned char* color = new unsigned char[4];
			size_t byteOffset = 0;
			size_t vertexByteSize = sizeof ( float ) *3 + sizeof ( unsigned char ) *4;
			BYTE* data = new BYTE[vertexByteSize*noTotalTriangles*3];
			for (int i = 0; i < noTotalTriangles; ++i) {
				pose[0] =triangleArray[i].p2.p.x;
				pose[1] =triangleArray[i].p2.p.y;
				pose[2] =triangleArray[i].p2.p.z;
				color[0] =triangleArray[i].p2.c.z;
				color[1] =triangleArray[i].p2.c.y;
				color[2] =triangleArray[i].p2.c.x;
				color[3] = 1;
				memcpy ( &data[byteOffset], pose, sizeof ( float ) *3 );
				byteOffset += sizeof ( float ) *3;
				memcpy ( &data[byteOffset], color, sizeof ( unsigned char ) *4 );
				byteOffset += sizeof ( unsigned char ) *4;

				pose[0] =triangleArray[i].p1.p.x;
				pose[1] =triangleArray[i].p1.p.y;
				pose[2] =triangleArray[i].p1.p.z;
				color[0] =triangleArray[i].p1.c.z;
				color[1] =triangleArray[i].p1.c.y;
				color[2] =triangleArray[i].p1.c.x;
				memcpy ( &data[byteOffset], pose, sizeof ( float ) *3 );
				byteOffset += sizeof ( float ) *3;
				memcpy ( &data[byteOffset], color, sizeof ( unsigned char ) *4 );
				byteOffset += sizeof ( unsigned char ) *4;

				pose[0] =triangleArray[i].p0.p.x;
				pose[1] =triangleArray[i].p0.p.y;
				pose[2] =triangleArray[i].p0.p.z;
				color[0] =triangleArray[i].p0.c.z;
				color[1] =triangleArray[i].p0.c.y;
				color[2] =triangleArray[i].p0.c.x;
				memcpy ( &data[byteOffset], pose, sizeof ( float ) *3 );
				byteOffset += sizeof ( float ) *3;
				memcpy ( &data[byteOffset], color, sizeof ( unsigned char ) *4 );
				byteOffset += sizeof ( unsigned char ) *4;

        	}	
			file.write ( ( const char* ) data, byteOffset );
			delete[] data;
			for ( size_t i = 0; i < m_FaceIndicesVertices.size(); i++ )
			{
				unsigned char numFaceIndices = ( unsigned char ) m_FaceIndicesVertices[i].size();
				file.write ( ( const char* ) &numFaceIndices, sizeof ( unsigned char ) );
				file.write ( ( const char* ) &m_FaceIndicesVertices[i][0], numFaceIndices*sizeof ( unsigned int ) );
			}
			file.close();
		}


		Triangle GetTriangle(int index)
		{
			Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);		
			return triangleArray[index];
		}

		void SwapAndClear()
		{
			cpu_triangles = new DWIO::MemoryBlock<Triangle>(noTotalTriangles, MEMORYDEVICE_CPU);
			//将顶点信息从gpu转到cpu上 cudaMemcpyDeviceToHost
			cudaMemcpy(cpu_triangles->GetData(MEMORYDEVICE_CPU),triangles->GetData(memoryType),sizeof(Triangle)*noTotalTriangles,cudaMemcpyDeviceToHost);
			triangles->Clear();
		}
		//将生成的临时的cpu_triangles删除
		void ClearCpuTriangles()
		{
			delete cpu_triangles;
		}


		~ITMMesh()
		{
			delete triangles;
		}

		// Suppress the default copy constructor and assignment operator
		ITMMesh(const ITMMesh&);
		ITMMesh& operator=(const ITMMesh&);
	};
}
