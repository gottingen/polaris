// Copyright 2024 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <polaris/internal/doubly_buffered_data.h>

#include <gtest/gtest.h>

namespace polaris {
namespace testing {

struct Foo {
  Foo() : x(0) {}
  int x;
};

bool Incr(Foo& f) {
  f.x++;
  return true;
}

bool AddN(Foo& f, int n) {
  f.x += n;
  return true;
}

bool AddMN(Foo& f, int m, int n) {
  f.x += m;
  f.x += n;
  return true;
}

class DBDTest : public ::testing::Test {};

TEST_F(DBDTest, read_modify) {
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.Modify(Incr);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(1, ptr->x);
    }
  }
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.Modify(AddN, 10);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(10, ptr->x);
    }
  }
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.Modify(AddMN, 10, 5);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(15, ptr->x);
    }
  }
}

bool BackGroundAddN(Foo& f, int n) {
  f.x += n;
  return false;
}

bool IncrWithForeGround(Foo& bg_f, const Foo& fg_f) {
  bg_f.x = fg_f.x;
  bg_f.x++;
  return true;
}

bool AddNWithForeGround(Foo& bg_f, const Foo& fg_f, int n) {
  bg_f.x = fg_f.x;
  bg_f.x += n;
  return true;
}

bool AddMNWithForeGround(Foo& bg_f, const Foo& fg_f, int m, int n) {
  bg_f.x = fg_f.x;
  bg_f.x += m;
  bg_f.x += n;
  return true;
}

TEST_F(DBDTest, read_modify_with_foreground) {
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    // 前台不会修改，后台变为3
    d.Modify(BackGroundAddN, 3);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.ModifyWithForeground(IncrWithForeGround);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(1, ptr->x);
    }
  }
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    // 前台不会修改，后台变为3
    d.Modify(BackGroundAddN, 3);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.ModifyWithForeground(AddNWithForeGround, 10);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(10, ptr->x);
    }
  }
  {
    DoublyBufferedData<Foo> d;
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    // 前台不会修改，后台变为3
    d.Modify(BackGroundAddN, 3);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(0, ptr->x);
    }
    d.ModifyWithForeground(AddMNWithForeGround, 10, 5);
    {
      DoublyBufferedData<Foo>::ScopedPtr ptr;
      EXPECT_EQ(0, d.Read(&ptr));
      EXPECT_EQ(15, ptr->x);
    }
  }
}

}  // namespace testing
}  // namespace polaris
