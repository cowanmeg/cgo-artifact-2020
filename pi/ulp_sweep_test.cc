/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ulp.h"
#include "ulp_neon.h"
#include "gtest/gtest.h"
#include "caffe2/core/timer.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#define REPEATS 10

namespace caffe2 {

void conv(const ConvArgs& args,
          const TensorCPU& X,
          const TensorCPU& W,
          const TensorCPU* b,
          TensorCPU* Y) {
  const auto N = X.dim32(0);
  const auto IH = X.dim32(1);
  const auto IW = X.dim32(2);
  const auto KH = W.dim32(1);
  const auto KW = W.dim32(2);
  const auto IC = W.dim32(3);
  Y->Resize(X.dim32(0),
            (X.dim32(1) - KH + args.pad_t + args.pad_b) / args.stride_h + 1,
            (X.dim32(2) - KW + args.pad_l + args.pad_r) / args.stride_w + 1,
            W.dim32(0));
  CHECK_EQ(W.dim32(3), X.dim32(3));
  const auto OH = Y->dim32(1);
  const auto OW = Y->dim32(2);
  const auto OC = Y->dim32(3);

  const auto* Xdata = X.data<float>();
  const auto* Wdata = W.data<float>();
  auto* Ydata = Y->mutable_data<float>();
  for (auto n = 0; n < N; ++n) {
    for (auto oh = 0; oh < OH; ++oh) {
      for (auto ow = 0; ow < OW; ++ow) {
        for (auto oc = 0; oc < OC; ++oc) {
          float acc = b ? b->data<float>()[oc] : 0.0;
          for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
              for (int ic = 0; ic < IC; ++ic) {
                if (kh + args.stride_h * oh - args.pad_t < 0 ||
                    kh + args.stride_h * oh - args.pad_t >= IH ||
                    kw + args.stride_w * ow - args.pad_l < 0 ||
                    kw + args.stride_w * ow - args.pad_l >= IW) {
                  continue;
                }
                const auto x =
                    Xdata[ic + IC * (kw + args.stride_w * ow - args.pad_l) +
                          IC * IW * (kh + args.stride_h * oh - args.pad_t) + n * IC * IW * IH];
                const auto w = Wdata[ic + IC * kw + IC * KW * kh + IC * KW * KH * oc];
                acc += x * w;
              }
            }
          }
          Ydata[oc + OC * ow + OC * OW * oh + n * OC * OW * OH] = acc;
        }
      }
    }
  }
}

int randInt(int a, int b) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

TensorCPU genTensor11(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen) > 0.5 ? -1.0 : 1.0;
  };
  return r;
}

TensorCPU genTensorUniform11(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(-5.0, 5.0);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = dis(gen);
  };
  return r;
}

TensorCPU genTensor0123(std::vector<TIndex> shape) {
  TensorCPU r;
  r.Resize(shape);

  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<float> dis(0.1, 3.9);

  for (auto i = 0; i < r.size(); ++i) {
    r.mutable_data<float>()[i] = std::floor(dis(gen));
  };
  return r;
}

void ConvTest2b1b(int IC, int KH, int KW, int H, int W, int OC, int N, ConvArgs args, std::ofstream &f) {
  args.stride_h = std::min(args.stride_h, KH);
  args.stride_w = std::min(args.stride_w, KW);
  args.pad_l = std::min(args.pad_l, KW - 1);
  args.pad_r = std::min(args.pad_r, KW - 1);
  args.pad_t = std::min(args.pad_t, KH - 1);
  args.pad_b = std::min(args.pad_b, KH - 1);

  LOG(INFO) << "IC: " << IC << ", KH: " << KH << ", KW: " << KW << ", H: " << H << ", W: " << W
            << ", OC: " << OC << ", N: " << N << ", pad_l: " << args.pad_l
            << ", pad_r: " << args.pad_r << ", pad_t: " << args.pad_t << ", pad_b: " << args.pad_b
            << ", stride_h: " << args.stride_h << ", stride_w: " << args.stride_w;
  auto X = genTensor0123({N, H, W, IC});
  auto W_ = genTensor11({OC, KH, KW, IC});
  auto bias = genTensorUniform11({OC});
  TensorPrinter tp;
  tp.PrintMeta(X);
  tp.PrintMeta(W_);
  tp.PrintMeta(bias);
  TensorCPU Y, YQ, Y2b1b, YOP;


  {
    Workspace ws;
    auto state = create2b1bConvState(&ws, W_, &bias);
    // Ignore first    
    run2b1bConvGeneric(state.get(), args, X, &Y2b1b);
    float times[REPEATS];
    for (int i = 0; i < REPEATS; i++) {
      Workspace ws;
      auto state = create2b1bConvState(&ws, W_, &bias);
      times[i] = run2b1bConvGeneric(state.get(), args, X, &Y2b1b);
    }
    float avg_time = 0.0;
    for (int i = 0; i < REPEATS; i++) {
       avg_time += times[i];
       f << times[i];
       if (i < (REPEATS - 1))
           f << ",";
    }
    f << "\n";
    avg_time /= static_cast<float>(REPEATS);
    LOG(INFO) << "AVERAGE TIME " << avg_time << std::endl;
  }
}

ConvArgs ca(size_t pad = 0, size_t stride = 1) {
  ConvArgs r;
  r.pad_l = pad;
  r.pad_r = pad;
  r.pad_t = pad;
  r.pad_b = pad;
  r.stride_w = stride;
  r.stride_h = stride;
  return r;
}

TEST(QConv, 2b1bConvTest) {
  //Sweep test
  std::ofstream f;
  f.open("raw_pytorch_bitserial_conv2d_a2w1_single.csv");
  // Resnet Layers 2-12
  ConvTest2b1b(64, 3, 3, 56, 56, 64, 1, ca(1,1), f);
  ConvTest2b1b(64, 1, 1, 56, 56, 64, 1, ca(0, 1), f);
  ConvTest2b1b(64, 3, 3, 56, 56, 128, 1, ca(1, 2), f);
  ConvTest2b1b(64, 1, 1, 56, 56, 128, 1, ca(0, 2), f);
  ConvTest2b1b(128, 3, 3, 28, 28, 128, 1, ca(1, 1), f);
  ConvTest2b1b(128, 3, 3, 28, 28, 256, 1, ca(1, 2), f);
  ConvTest2b1b(128, 1, 1, 28, 28, 256, 1, ca(0, 2), f);
  ConvTest2b1b(256, 3, 3, 14, 14, 256, 1, ca(1, 1), f);
  ConvTest2b1b(256, 3, 3, 14, 14, 512, 1, ca(1, 2), f);
  ConvTest2b1b(256, 1, 1, 14, 14, 512, 1, ca(0, 2), f);
  ConvTest2b1b(512, 3, 3, 7, 7, 512, 1, ca(1, 1), f);
  f.close();
}

} // namespace caffe2
