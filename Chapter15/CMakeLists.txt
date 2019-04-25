# cmake for OpenCV Cookbook 3rd edition chapter 15
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( faceDetection face_detection.cpp)
add_executable( objectDetection object_detection_yolov3.cpp)

# link libraries
target_link_libraries( faceDetection ${OpenCV_LIBS})
target_link_libraries( objectDetection ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (DATA ${CMAKE_SOURCE_DIR}/Chapter15/data/coco.names
            ${CMAKE_SOURCE_DIR}/Chapter15/data/deploy.prototxt
            ${CMAKE_SOURCE_DIR}/Chapter15/data/res10_300x300_ssd_iter_140000.caffemodel
            ${CMAKE_SOURCE_DIR}/Chapter15/data/yolov3.cfg
            # ${CMAKE_SOURCE_DIR}/Chapter15/data/yolov3.weights
            ${CMAKE_SOURCE_DIR}/Chapter15/data/test.jpg
)
FILE(COPY ${DATA} DESTINATION ./data/)
MESSAGE(STATUS "Downloading YOLOV3 weights")
FILE(DOWNLOAD https://pjreddie.com/media/files/yolov3.weights ./Chapter15/data/yolov3.weights SHOW_PROGRESS)

