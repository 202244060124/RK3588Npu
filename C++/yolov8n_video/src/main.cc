// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include "postprocess.h"
#include "rknn_api.h"
using namespace std;


static void DumpTensorAttrData(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
        get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double GetUsTime(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}

static unsigned char* LoadFileData(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char* LoadModel(const char* filename, int* model_size)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = LoadFileData(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int SaveFloatData(const char* file_name, float* output, int element_size)
{
    FILE* fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++) {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
    int status = 0;
    char* model_path = NULL;
    rknn_context ctx;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time;
    struct timeval stop_time;
    int ret;

    if (argc != 3) {
        printf("Usage: %s <rknn model> <video path> \n", argv[0]);
        return -1;
    }

    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

    model_path = (char*)argv[1];

    std::string video_path = argv[2];

    cv::namedWindow("Image Window");

    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char* model_data = LoadModel(model_path, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        DumpTensorAttrData(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        DumpTensorAttrData(&(output_attrs[i]));
    }

    int input_channel = 3;
    int input_width = 0;
    int input_height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        input_channel = input_attrs[0].dims[1];
        input_width = input_attrs[0].dims[2];
        input_height = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        input_width = input_attrs[0].dims[1];
        input_height = input_attrs[0].dims[2];
        input_channel = input_attrs[0].dims[3];
    }

    printf("model input input_height = %d, input_width = %d, input_channel = %d\n", input_height, input_width, input_channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_width * input_height * input_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to dst resulotion
    void* resize_buf = malloc(input_height * input_width * input_channel);
    if (resize_buf == nullptr) {
        printf("malloc input buf failed!");
        return -1;
    }
    inputs[0].buf = resize_buf;

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 0;
    }

    cv::Mat orig_img;
    cv::Mat img;
    cv::Mat resize_img;
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        printf("capture device failed to open!");
        cap.release();
        exit(-1);
    }

    if (!cap.read(orig_img)) {
        printf("Capture read error");
    }
    cv::namedWindow("Driver", cv::WINDOW_NORMAL);
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;
    printf("img input_width = %d, img input_height = %d\n", img_width, img_height);

    // post process
    float scale_w = (float)input_width / img_width;
    float scale_h = (float)input_height / img_height;

    detect_result_group_t detect_result_group;
    vector<float> out_scales;
    vector<int32_t> out_zps;
    char text[256];


    while (1) {
        if (!cap.read(orig_img)) {
            printf("Capture read error");
            break;
        }
        cv::resize(orig_img, resize_img, cv::Size(input_width, input_height));
        cv::cvtColor(resize_img, img, cv::COLOR_BGR2RGB);
        memset(inputs[0].buf, 0x00, input_height * input_width * input_channel);
        memcpy(inputs[0].buf, (void*)img.data, input_height * input_width * input_channel);

        gettimeofday(&start_time, NULL);
        rknn_inputs_set(ctx, io_num.n_input, inputs);

        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        gettimeofday(&stop_time, NULL);
        printf("once run use %f ms\n", (GetUsTime(stop_time) - GetUsTime(start_time)) / 1000);

        out_scales.clear();
        out_zps.clear();
        for (int i = 0; i < io_num.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }
        memset(&detect_result_group, 0, sizeof(detect_result_group_t));
        post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, input_height, input_width,
            box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t* det_result = &(detect_result_group.results[i]);
            printf("object:%s:\n", det_result->name);
            printf("[%d %d %d %d] %f\n", det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                det_result->prop);
            text[0] = '\0';
            sprintf(text, "%s:%.1f", det_result->name, det_result->prop * 100);

            rectangle(orig_img, cv::Point(det_result->box.left, det_result->box.top),
                cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(255, 0, 0, 255), 2);
            putText(orig_img, text, cv::Point(det_result->box.left, det_result->box.top - 12), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 0, 255));
        }

        cv::imshow("Driver", orig_img);
        cv::waitKey(1);

        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    deinitPostProcess();
    // release
    ret = rknn_destroy(ctx);
    cv::destroyAllWindows();
    if (model_data) {
        free(model_data);
    }

    if (resize_buf) {
        free(resize_buf);
    }

    return 0;
}
