// Times the Orukeet/Parakeet ONNX encoder (deterministic synthetic input) and dumps its output.
#include "onnxruntime_cxx_api.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
  const int T = 1100; int threads = atoi(argv[2]);
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "check");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(threads);
  Ort::Session s(env, argv[1], so);
  std::vector<float> x(128 * T); srand(1234); for (auto& v : x) v = (rand() % 1000) / 500.f - 1.f;
  std::vector<int64_t> len{T}; int64_t xs[3] = {1, 128, T}; int64_t ls[1] = {1};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in[2] = {Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), xs, 3),
                      Ort::Value::CreateTensor<int64_t>(mem, len.data(), 1, ls, 1)};
  const char* inn[2] = {"audio_signal", "length"};
  auto names = s.GetOutputNames(); std::vector<const char*> on; for (auto& n : names) on.push_back(n.c_str());
  auto out = s.Run(Ort::RunOptions{nullptr}, inn, in, 2, on.data(), on.size());
  if (argc > 3) { auto n = out[0].GetTensorTypeAndShapeInfo().GetElementCount(); FILE* f = fopen(argv[3], "wb"); fwrite(out[0].GetTensorData<float>(), 4, n, f); fclose(f); }
  for (int i = 0; i < 2; i++) s.Run(Ort::RunOptions{nullptr}, inn, in, 2, on.data(), on.size());
  auto t0 = std::chrono::steady_clock::now(); const int n = 5;
  for (int i = 0; i < n; i++) s.Run(Ort::RunOptions{nullptr}, inn, in, 2, on.data(), on.size());
  printf("encoder threads=%d: %.0f ms/run\n", threads, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n);
}
