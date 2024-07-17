1、根据电脑的GPU选择安装对应的cuda版本
2、根据安装cuda版本安装对应的opencv、opencv-contrib
3、pangolin版本：0.6、eigen版本：3.4.0
4、编译：（编译前修改CMakeLists.txt中的对应的cuda编译版本为与自己电脑配置一致的版本）
	mkdir build && cd build
	cmake ..
	make -j12
	
