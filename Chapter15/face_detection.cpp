/**
This sample uses Single-Shot Detector 
(https://arxiv.org/abs/1512.02325) 
with ResNet-10 architecture to detect faces on camera/video/image.
More information about the training is available here: 
<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt
.caffemodel model's file is available here:
<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel
.prototxt file is available here: 
<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

// Set up configuration
float confidenceThreshold = 0.5;
String modelConfiguration = "data/deploy.prototxt";
String modelBinary = "data/res10_300x300_ssd_iter_140000.caffemodel";
const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

int main(int argc, char** argv)
{
    
    // Init DNN
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    
    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

    VideoCapture cap;
    if (argc==1)
    {
        cap = VideoCapture(0);
        if(!cap.isOpened())
        {
            cout << "Couldn't find  default camera" << endl;
            return -1;
        }
    }
    else
    {
        cap.open(argv[1]);
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << argv[1] << endl;
            return -1;
        }
    }

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera/video or read image

        if (frame.empty())
        {
            waitKey();
            break;
        }

        //! [Prepare blob]
        Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images
        
        //! [Set input blob]
        net.setInput(inputBlob, "data"); //set the network input
        
        //! [Make forward pass]
        Mat detection = net.forward("detection_out"); //compute output
        
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(frame, object, Scalar(0, 255, 0));

                stringstream ss;
                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = "Face: " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                      Size(labelSize.width, labelSize.height + baseLine)),
                          Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        imshow("detections", frame);
        if (waitKey(1) >= 0) break;
    }

    return 0;
} // main
