cmake_minimum_required(VERSION 3.16)
project(GestureReco)

set(CMAKE_CXX_STANDARD 14)
set(APP_NAME GestureReco)


# Import OpenCV3
# TODO modify to your own path
set(OPENCV_HOME "/usr/local/Cellar/opencv@3/3.4.10")
set(OpenCV_DIR ${OPENCV_HOME}/share/OpenCV/)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Compiler support Object C++ to call System API
add_compile_options(-x objective-c++)
set(CMAKE_EXE_LINKER_FLAGS "-framework Cocoa")

set(Tensorflow_HOME "/opt/tensorflow") # TODO modify to your own path

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/lib
)

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${Tensorflow_HOME}/bazel-tensorflow/external/com_google_absl
        ${Tensorflow_HOME}/bazel-tensorflow/external/gemmlowp
        ${Tensorflow_HOME}/bazel-tensorflow/external/flatbuffers/include
        ${Tensorflow_HOME}
)
add_library(tensorflowlite SHARED IMPORTED)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src src_files)

add_executable(
        ${APP_NAME} src/Entry.cpp
        ${src_files}
)
target_link_libraries(${APP_NAME} PRIVATE ${OpenCV_LIBS} -ltensorflowlite)

configure_file(model/BlazePalm.tflite model/BlazePalm.tflite COPYONLY)
configure_file(model/BlazePalmAnchors.csv model/BlazePalmAnchors.csv COPYONLY)