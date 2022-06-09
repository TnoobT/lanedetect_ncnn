#include "lane.hpp"

using namespace std;
using namespace cv; 

int main()
{
    string img_path = "../img/1.png";
    string param_path = "../models/mlsd_no_max_sigmoid_sim.param";
    string bin_path = "../models/mlsd_no_max_sigmoid_sim.bin";

    Mat src_img = imread(img_path);
    Mat image = src_img.clone();
    LaneDetect lanedetect(param_path,bin_path);
    std::vector<LaneDetect::Lanes> lanes = lanedetect.inference(image);
    
    return 0;




}