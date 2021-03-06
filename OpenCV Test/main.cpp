//
//  main.cpp
//  OpenCV Test
//
//  Created by Jorge Alejandro Ricaurte on 3/30/22.
//

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>

using namespace cv;

double vec_sum(std::vector<double> inputVec){
    double sum = 0;
    for (int i=0; i<inputVec.size(); i++) {
        sum+= inputVec[i];
    }
    return sum;
}

void calibrate_camera(Mat& cameraMatrix, Mat& distCoeffs, bool visualize=false, bool report=false){
    // Defining the dimensions of checkerboard
    int CHECKERBOARD[2]{8,6};
    
    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;
    
    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;
    
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {
        for(int j{0}; j<CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }
    
    
    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
    std::string path = "/Users/jorgericaurte/Documents/University/Research/Invertec Pendulum openCV/OpenCV Test/OpenCV Test/CalibrationImages/*.jpg";
    
    cv::glob(path, images);
    
    cv::Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners
    std::vector<cv::Point2f> corner_pts;
    bool success;
    
    // Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
        
        // Finding checker board corners
        // If desired number of corners are found in the image then success = true
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        
        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checker board
         */
        if(success)
        {
            cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
            
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        imwrite("/Users/jorgericaurte/Documents/University/Research/Invertec Pendulum openCV/OpenCV Test/OpenCV Test/CalibrationImages/Calibration_image_"+std::to_string(i)+".jpg", frame);
        if (visualize) {
            cv::imshow("Image",frame);
            
            cv::waitKey(0);
        }
        
    }
    if (visualize) {
        cv::destroyAllWindows();
    }
    
    
    cv::Mat R,T;
    
    /*
     * Performing camera calibration by
     * passing the value of known 3D points (objpoints)
     * and corresponding pixel coordinates of the
     * detected corners (imgpoints)
     */
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
    if (report) {
        std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
        std::cout << "distCoeffs : " << distCoeffs << std::endl;
        std::cout << "Rotation vector : " << R << std::endl;
        std::cout << "Translation vector : " << T << std::endl;
    }
}

void two_marker_angles(std::vector<std::vector<cv::Point2f>>& markerCorners, std::vector<double>& avg_angles, std::vector<double>& times_vec, double current_time){
    std::vector<double> angles;
    for (int i=0; i<markerCorners.size()-1; i++) {
        angles = {};
        for(int j=0; j<markerCorners[i].size(); j++){
            double x_1=markerCorners[i][j].x, x_2 = markerCorners[i+1][j].x,
            y_1 = markerCorners[i][j].y, y_2 = markerCorners[i+1][j].y;
            angles.push_back(atan2(abs(y_2-y_1),x_2-x_1));
        }
        avg_angles.push_back(vec_sum(angles)/double(angles.size()));
        times_vec.push_back(current_time);
    }
}

std::vector<double> markerCenter(std::vector<std::vector<cv::Point2f>> markerCorners, std::vector<double>& times_vec, double current_time){
    
    std::vector<double>  centerCoords;
    for (int j=0; j<markerCorners.size(); j++) {
        
        double xCenter=0, yCenter=0;
        for (int i=0; i<markerCorners[j].size(); i++) {
            xCenter+=markerCorners[j][i].x;
            yCenter+=markerCorners[j][i].y;
        }
        xCenter/=4;
        centerCoords.push_back(xCenter);
        yCenter/=4;
        centerCoords.push_back(yCenter);
    }
    times_vec.push_back(current_time);
    return centerCoords;
}

int main(){
    bool calc_angles = false, save=false, view=false;
    cv::Mat cameraMatrix,distCoeffs;
    calibrate_camera(cameraMatrix, distCoeffs);
    VideoCapture cap("/Users/jorgericaurte/Documents/University/Research/Invertec Pendulum openCV/OpenCV Test/OpenCV Test/coordinate_trial_2.mp4");
    
    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    // Default resolutions of the frame are obtained.The default resolutions are system dependent.
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    VideoWriter video("/Users/jorgericaurte/Documents/University/Research/Invertec Pendulum openCV/OpenCV Test/OpenCV Test/outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, Size(frame_width,frame_height));
    if (!save) {
        video.release();
    }
    
    
    std::vector<double> avgAngles, timesVec;
    std::vector<std::vector<double>> centerCoordVec;
    double current_time = 0;
    
    while(1){
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        
        // Display the resulting frame
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        if (markerCorners.size()>1 && calc_angles) {
            two_marker_angles(markerCorners, avgAngles, timesVec, current_time);
        }
        else if(markerCorners.size()>1 && !calc_angles){
            centerCoordVec.push_back(markerCenter(markerCorners, timesVec, current_time));
        }
        current_time += 1.0/30.0;
        cv::Mat outputFrame = frame.clone();
        if (markerIds.size() > 0 && view) {
            cv::aruco::drawDetectedMarkers(outputFrame, markerCorners, markerIds);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
            // draw axis for each marker
                for(int i=0; i<markerIds.size(); i++)
                    cv::drawFrameAxes(outputFrame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
            cv::aruco::drawDetectedMarkers(outputFrame, markerCorners, markerIds);
        }
        if (view) {
            imshow( "Frame", outputFrame );
        }
        // Write the frame into the file 'outcpp.avi'
        if (save) {
            video.write(outputFrame);
        }
    }
    
    // When everything done, release the video capture object
    cap.release();
    if(save) video.release();
    
    // Closes all the frames
    destroyAllWindows();
    std::ofstream outputFile;
    outputFile.open("/Users/jorgericaurte/Documents/University/Research/Invertec Pendulum openCV/OpenCV Test/OpenCV Test/coordinates_2.csv");
    for (int i=0; i<timesVec.size(); i++) {
        if (calc_angles) {
            avgAngles[i]*=180/M_PI;
            outputFile<<timesVec[i]<<","<<avgAngles[i]<<"\n";
        }
        else if(!calc_angles){
            outputFile<<timesVec[i];
            for (int j=0; j<centerCoordVec[i].size(); j++) {
                outputFile<<","<<centerCoordVec[i][j];
            }
            outputFile<<"\n";
        }
    }
    outputFile.close();
    std::cout<<"Done\n";
    return 0;
}
