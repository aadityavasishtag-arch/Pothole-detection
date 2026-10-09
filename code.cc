
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // Step 1: Load the road image
    string filename = (argc > 1) ? argv[1] : "road.jpg";
    Mat image = imread(filename);

    if (image.empty()) {
        cout << "Error: Could not load road image!" << endl;
        cout << "Place road.jpg in the program folder or provide its path."
             << endl;
        return 1;
    }

    // Step 2: Convert image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Step 3: Reduce noise
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // Step 4: Apply Black Hat morphological operation
    // Highlights dark regions surrounded by relatively brighter areas
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE, Size(25, 25)
    );

    Mat blackhat;
    morphologyEx(blurred, blackhat, MORPH_BLACKHAT, kernel);

    // Step 5: Threshold the image
    Mat binary;
    threshold(
        blackhat, binary, 0, 255,
        THRESH_BINARY | THRESH_OTSU
    );

    // Step 6: Remove small noise and join nearby regions
    Mat smallKernel = getStructuringElement(
        MORPH_ELLIPSE, Size(3, 3)
    );

    morphologyEx(binary, binary, MORPH_OPEN, smallKernel);
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // Step 7: Find contours
    vector<vector<Point>> contours;
    findContours(
        binary, contours, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    int potholeCount = 0;
    Mat result = image.clone();

    // Step 8: Filter contours and mark suspected potholes
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // Ignore very small regions and huge background regions
        if (area < 300 || area > 50000) {
            continue;
        }

        Rect box = boundingRect(contours[i]);

        double aspectRatio =
            static_cast<double>(box.width) / box.height;

        // Reject extremely thin or elongated regions
        if (aspectRatio < 0.3 || aspectRatio > 3.5) {
            continue;
        }

        potholeCount++;

        // Draw a red bounding box
        rectangle(
            result, box, Scalar(0, 0, 255), 2
        );

        putText(
            result,
            "Possible Pothole",
            Point(box.x, max(20, box.y - 8)),
            FONT_HERSHEY_SIMPLEX,
            0.55,
            Scalar(0, 0, 255),
            2
        );
    }

    // Step 9: Display the total count
    string countText =
        "Suspected Potholes: " + to_string(potholeCount);

    putText(
        result, countText, Point(20, 35),
        FONT_HERSHEY_SIMPLEX, 0.8,
        Scalar(0, 255, 0), 2
    );

    // Step 10: Save and display the result
    imwrite("pothole_result.jpg", result);
    imwrite("pothole_mask.jpg", binary);

    cout << "Image processed successfully!" << endl;
    cout << "Suspected potholes: " << potholeCount << endl;
    cout << "Output saved as pothole_result.jpg" << endl;

    imshow("Original Road Image", image);
    imshow("Detected Potholes", result);
    imshow("Image Processing Mask", binary);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
