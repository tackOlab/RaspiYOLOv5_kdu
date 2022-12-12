cmake_minimum_required(VERSION 3.13)
project (TEST LANGUAGES CXX DESCRIPTION "OpenCV examples")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_MULTITHREAD "enable multi-thread" OFF)

if (ENABLE_MULTITHREAD)
	find_package(Threads)
endif()

find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA libcamera)
if (LIBCAMERA_FOUND)
	message(STATUS "libcamera library found:")
	message(STATUS "    version: ${LIBCAMERA_VERSION}")
	message(STATUS "    libraries: ${LIBCAMERA_LINK_LIBRARIES}")
	message(STATUS "    include path: ${LIBCAMERA_INCLUDE_DIRS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_LIBCAMERA")
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^arm64")
	if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
		message(STATUS "Add compiler option -mcpu=cortex-a72")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a72 -mtune=native")
	endif()
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ext/OpenHTJ2K)
include_directories(ext/OpenHTJ2K/src/core/interface)

if (LIBCAMERA_FOUND)
	include_directories(${CMAKE_SOURCE_DIR} ${LIBCAMERA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
	add_executable(yolo main.cpp LibCamera.cpp ohtj2k_codec.cpp)
	target_include_directories(yolo PUBLIC ${LIBCAMERA_INCLUDE_DIRS})
	target_link_libraries(yolo open_htj2k ${OpenCV_LIBS} ${LIBCAMERA_LINK_LIBRARIES})
else()
	include_directories(${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
	add_executable(yolo main.cpp ohtj2k_codec.cpp)
	target_link_libraries(yolo open_htj2k ${OpenCV_LIBS})
endif()

if (Threads_FOUND)
 target_link_libraries(yolo PUBLIC ${CMAKE_THREAD_LIBS_INIT})
endif()

