#include "lane.hpp"

LaneDetect::LaneDetect(std::string param_path,std::string bin_path)
{
    m_net = new ncnn::Net();
    m_net->opt.use_vulkan_compute = false; //hasGPU && useGPU;  // gpu
    m_net->opt.use_fp16_arithmetic = true;
    m_net->opt.use_fp16_packed = true;
    m_net->opt.use_fp16_storage = true;
    m_net->load_param(param_path.c_str());
    m_net->load_model(bin_path.c_str());

}


LaneDetect::~LaneDetect()
{   
    m_net->clear();
    delete m_net;
}

inline int LaneDetect::clip(float value)
{
    if (value > 0 && value < m_input_size)
        return int(value);
    else if (value < 0)
        return 1;
    else
        return m_input_size - 1;

}

void LaneDetect::showImg(const cv::Mat& img,std::vector<LaneDetect::Lanes> Lanes)
{
    cv::Mat tmp_img = img.clone();
    int h = tmp_img.rows;
    int w = tmp_img.cols;
    
    for(auto line:Lanes)
    {
        float x1 = line.x1;
        float y1 = line.y1;
        float x2 = line.x2;
        float y2 = line.y2;
        x1 = clip(w * x1 / (m_input_size / 2));
        x2 = clip(w * x2 / (m_input_size / 2));
        y1 = clip(h * y1 / (m_input_size / 2));
        y2 = clip(h * y2 / (m_input_size / 2));

        cv::line(tmp_img,cv::Point(x1,y1),cv::Point(x2,y2), cv::Scalar(0, 0, 255));
    }

    cv::imshow("img",tmp_img);
    cv::waitKey(0);
    return ;
}



std::vector<LaneDetect::Lanes> LaneDetect::decodeHeatmap(const float* hm)
{   
    // 线段中心点(256*256),线段偏移(4*256*256)
    const float*  displacement = hm+m_hm_size*m_hm_size;
    // exp(center,center);
    std::vector<float> center;
    for (int i = 0;i < m_hm_size*m_hm_size; i++)
    {
        center.push_back( 1/(exp(-hm[i]) + 1) ); // mlsd.mnn原始需要1/(exp(-hm[i]) + 1)
    }
    center.resize(m_hm_size*m_hm_size);

    std::vector<int> index(center.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (center[a] > center[b]); // 从大到小排序
        }
    );
    
    for (int i = 0; i < index.size(); i++)
    {
        int yy = index[i]/m_hm_size; // 除以宽得行号
        int xx = index[i]%m_hm_size; // 取余宽得列号
        Lanes Lane;
        Lane.x1 = xx + displacement[index[i] + 0*m_hm_size*m_hm_size];
        Lane.y1 = yy + displacement[index[i] + 1*m_hm_size*m_hm_size];
        Lane.x2 = xx + displacement[index[i] + 2*m_hm_size*m_hm_size];
        Lane.y2 = yy + displacement[index[i] + 3*m_hm_size*m_hm_size];
        Lane.lens = sqrt(pow(Lane.x1 - Lane.x2,2) + pow(Lane.y1 - Lane.y2,2));
        Lane.conf = center[index[i]];

        if (center[index[i]] > m_score_thresh && m_lanes.size() < m_top_k)
        {
            if ( Lane.lens > m_min_len)
                m_lanes.push_back(Lane);
        }
        else
            break;
    }
    
    return m_lanes;

}

void LaneDetect::processImg(const cv::Mat& image,ncnn::Mat& in)
{
    int img_w = image.cols;
    int img_h = image.rows;
    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, img_w, img_h);
    in.substract_mean_normalize(m_mean_vals, m_norm_vals);
}

const float* LaneDetect::inference(const cv::Mat& img)
{
    cv::Mat preImage = img.clone();
    cv::resize(preImage,preImage,cv::Size(m_input_size,m_input_size));
    cv::cvtColor(preImage,preImage,cv::COLOR_BGR2RGB);
    ncnn::Mat input;
    processImg(preImage, input); // 图片预处理
    auto ex = m_net->create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    ex.set_vulkan_compute(this->hasGPU);
#endif
    ex.input("input", input);
    ncnn::Mat out;
    ex.extract("output", out); //输出
    
    return out;
}