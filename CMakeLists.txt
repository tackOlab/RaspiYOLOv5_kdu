cmake_minimum_required(VERSION 3.13)
project (TEST LANGUAGES CXX DESCRIPTION "Network surveillance camera system based on HTJ2K and Raspberry Pi with ArduCAM autofocus 16MP")
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
		message(STATUS "Add compiler option -mcpu=cortex-a76")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a76 -mtune=native -march=armv8.1-a")
	endif()
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual -Wno-array-parameter -Wno-unused-but-set-variable -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ext/kakadujs)
include_directories(ext/kakadujs/src)

# For Raspberry Pi with Pi Camera
if (LIBCAMERA_FOUND)
	add_executable(yolo main.cpp LibCamera.cpp)
	target_include_directories(yolo PRIVATE ${LIBCAMERA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(yolo kakadujs ${OpenCV_LIBS} ${LIBCAMERA_LINK_LIBRARIES})
	if (Threads_FOUND)
		target_link_libraries(yolo PUBLIC ${CMAKE_THREAD_LIBS_INIT})
	endif()
endif()

# For still pictures
add_executable(yolo_still main_still.cpp)
target_include_directories(yolo_still PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_still ${OpenCV_LIBS})

# For general vides inputs
add_executable(yolo_vid main_vid.cpp)
target_include_directories(yolo_vid PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_vid kakadujs ${OpenCV_LIBS})



