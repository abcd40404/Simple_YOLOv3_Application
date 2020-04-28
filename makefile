all:
	g++ -Wall -o yolo -std=c++11 yolo_app.cpp -Iinclude -L. -ldarknet -lopencv_core -lopencv_highgui -lopencv_imgproc

clean:
	rm yolo
