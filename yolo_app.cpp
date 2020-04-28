#include <string>
#include <vector>
#include <fstream>
#include "imgProcess.h"
extern "C"{
#include "darknet.h"
}

using namespace std;
using namespace cv;

int main(){
    float thresh=0.5;
    float nms=0.45;
    string cfgfile = "cfg/myyolov3.cfg";
    string weightfile = "./yolov3.weights";

    network *net = load_network((char*)cfgfile.c_str(),(char*)weightfile.c_str(),0);
    set_batch_network(net, 1);
    vector<string> classNamesVec;
    ifstream classNamesFile("cfg/crossing.names");
         
    if (classNamesFile.is_open()){
        string className = "";
        while (getline(classNamesFile, className))
        classNamesVec.push_back(className);                
    }
    int classes = classNamesVec.size();
    cv::Mat frame, rgbImg;
    frame = cv::imread("crossing.png");

	cvtColor(frame, rgbImg, cv::COLOR_BGR2RGB);
 
	float* srcImg;
	size_t srcSize=rgbImg.rows*rgbImg.cols*3*sizeof(float);
	srcImg=(float*)malloc(srcSize);

	imgConvert(rgbImg,srcImg);

	float* resizeImg;
	size_t resizeSize=net->w*net->h*3*sizeof(float);
	resizeImg=(float*)malloc(resizeSize);
	imgResize(srcImg,resizeImg,frame.cols,frame.rows,net->w,net->h);

	network_predict(net, resizeImg);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, rgbImg.cols, rgbImg.rows, thresh, 0.5, 0,1, &nboxes);
     
    if(nms){
        do_nms_sort(dets, nboxes, classes, nms);
    }
	vector<cv::Rect>boxes;
	boxes.clear();
	vector<int>classNames;

	for (int i = 0; i < nboxes; i++){
		bool flag = 0;
		int classIdx;
		for(int j = 0; j < classes; j++){
			if(dets[i].prob[j] > thresh){
				if(!flag){
					flag = 1;
					classIdx = j;
				}
                printf("%s: %f\n", classNamesVec[j].c_str(), dets[i].prob[j]);
			}
		}
		if(flag){
			int left = (dets[i].bbox.x - dets[i].bbox.w / 2.)*frame.cols;
			int right = (dets[i].bbox.x + dets[i].bbox.w / 2.)*frame.cols;
			int top = (dets[i].bbox.y - dets[i].bbox.h / 2.)*frame.rows;
			int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.)*frame.rows;

			if (left < 0)
				left = 0;
			if (right > frame.cols - 1)
				right = frame.cols - 1;
			if (top < 0)
				top = 0;
			if (bot > frame.rows - 1)
				bot = frame.rows - 1;

            cv::Rect box(left, top, fabs(left - right), fabs(top - bot));
			boxes.push_back(box);
			classNames.push_back(classIdx);
		}
	}
	free_detections(dets, nboxes);

	for(int i = 0; i < boxes.size(); i++){
		int offset = classNames[i]*123457 % 80;
		float red = 255*get_color(2,offset,80);
		float green = 255*get_color(1,offset,80);
		float blue = 255*get_color(0,offset,80);

		rectangle(frame, boxes[i], Scalar(blue,green,red), 2);
        cout << "Mask: " << boxes[i].x << " " << boxes[i].y << endl;

		String label = String(classNamesVec[classNames[i]]);
		int baseLine = 0;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		putText(frame, label, Point(boxes[i].x, boxes[i].y + labelSize.height),
				FONT_HERSHEY_SIMPLEX, 1, Scalar(red, blue, green),2);
	}
	imshow("video",frame);
    cv::waitKey(0);


	free(srcImg);
	free(resizeImg);
    free_network(net);
    return 0;
}
