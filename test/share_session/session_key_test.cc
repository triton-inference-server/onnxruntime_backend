// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <iostream>
#include <map>
#include <stdexcept>

#include "session_key.h"

namespace {

void
Check(bool condition, const char* message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void
TestSessionKey()
{
  using triton::backend::onnxruntime::SessionKey;
  constexpr auto gpu = TRITONSERVER_INSTANCEGROUPKIND_GPU;
  constexpr auto cpu = TRITONSERVER_INSTANCEGROUPKIND_CPU;
  const SessionKey gpu0{"model.with_underscores_12", gpu, 0};
  const SessionKey gpu1{"model.with_underscores_12", gpu, 1};
  const SessionKey cpu0{"model.with_underscores_12", cpu, 0};
  const SessionKey other_group{"model.with_underscores_13", gpu, 0};

  // Insert GPU 1 first: initialization order must not affect device isolation.
  std::map<SessionKey, int> sessions{{gpu1, 1}};
  Check(sessions.find(gpu0) == sessions.end(), "GPU 0 reused GPU 1's session");
  sessions.emplace(gpu0, 2);
  Check(sessions.find(cpu0) == sessions.end(), "CPU reused a GPU session");
  sessions.emplace(cpu0, 3);
  Check(
      sessions.find(other_group) == sessions.end(),
      "Another instance group reused a session");
  sessions.emplace(other_group, 4);

  const SessionKey same_gpu0{"model.with_underscores_12", gpu, 0};
  Check(sessions.at(same_gpu0) == 2, "Same-device instances did not share");
  Check(
      !sessions.emplace(same_gpu0, 5).second,
      "Same-device instance created a duplicate cache entry");
  Check(sessions.at(gpu1) == 1, "GPU 1's session was overwritten");
  Check(sessions.at(cpu0) == 3, "CPU session was overwritten");
  Check(sessions.at(other_group) == 4, "Other group session was overwritten");
  Check(sessions.size() == 4, "Incorrect session count");
}

}  // namespace

int
main()
{
  try {
    TestSessionKey();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Session key tests passed\n";
  return 0;
}
