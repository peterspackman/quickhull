#include "math_utils.hpp"
#include "quickhull.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

namespace quickhull {

namespace tests {

using FloatType = float;
using vec3 = Eigen::Matrix<FloatType, 3, 1>;

static void assertSameValue(FloatType a, FloatType b) {
  assert(std::abs(a - b) < 0.0001f);
}

// Setup RNG
static std::mt19937 rng;
static std::uniform_real_distribution<> dist(0, 1);

FloatType rnd(FloatType from, FloatType to) {
  return from + (FloatType)dist(rng) * (to - from);
};

static void testVector3() {
  typedef Eigen::Matrix<FloatType, 3, 1> vec3;
  vec3 a(1, 0, 0);
  vec3 b(1, 0, 0);

  vec3 c = b * (a.dot(b) / b.squaredNorm());
  assert((c - a).norm() < 0.00001f);

  a = vec3(1, 1, 0);
  b = vec3(1, 3, 0);
  c = a * (b.dot(a) / a.squaredNorm());
  assert((c - vec3(2, 2, 0)).norm() < 0.00001f);
}

template <typename T>
static Eigen::Matrix<T, 3, Eigen::Dynamic>
createSphere(T radius, size_t M,
             Eigen::Matrix<T, 3, 1> offset = Eigen::Matrix<T, 3, 1>(0, 0, 0)) {
  Eigen::Matrix<T, 3, Eigen::Dynamic> pc(3, M + 1);
  const T pi = 3.14159f;
  for (int i = 0; i <= M; i++) {
    FloatType y = std::sin(pi / 2 + (FloatType)i / (M)*pi);
    FloatType r = std::cos(pi / 2 + (FloatType)i / (M)*pi);
    FloatType K =
        FloatType(1) -
        std::abs((FloatType)((FloatType)i - M / 2.0f)) / (FloatType)(M / 2.0f);
    const size_t pcount = (size_t)(1 + K * M + FloatType(1) / 2);
    for (size_t j = 0; j < pcount; j++) {
      FloatType x =
          pcount > 1 ? r * std::cos((FloatType)j / pcount * pi * 2) : 0;
      FloatType z =
          pcount > 1 ? r * std::sin((FloatType)j / pcount * pi * 2) : 0;
      pc.col(i) = Eigen::Matrix<T, 3, 1>(x + offset.x(), y + offset.y(),
                                         z + offset.z());
    }
  }
  return pc;
}

static void sphereTest() {
  QuickHull<FloatType> qh;
  FloatType y = 1;
  for (;;) {
    auto pc = createSphere<FloatType>(1, 100,
                                      Eigen::Matrix<FloatType, 3, 1>(0, y, 0));
    auto hull = qh.getConvexHull(pc, true);
    y *= 15;
    y /= 10;
    if (hull.indices().size() == 12) {
      break;
    }
  }

  // Test worst case scenario: more and more points on the unit sphere. All
  // points should be part of the convex hull, as long as we can make epsilon
  // smaller without running out of numerical accuracy.
  size_t i = 1;
  FloatType eps = 0.002f;
  for (;;) {
    auto pc =
        createSphere<FloatType>(1, i, Eigen::Matrix<FloatType, 3, 1>(0, 0, 0));
    auto hull = qh.getConvexHull(pc, true, eps);
    if (qh.getDiagnostics().m_failedHorizonEdges) {
      // This should not happen
      assert(false);
      break;
    }
    if (pc.size() == hull.vertices().size()) {
      // Fine, all the points on unit sphere do belong to the convex mesh.
      i += 1;
    } else {
      eps *= 0.5f;
    }
    if (i == 100) {
      break;
    }
  }
}

static void testPlanarCase() {
  QuickHull<FloatType> qh;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc(3, 4);
  pc.col(0) =
      Eigen::Matrix<FloatType, 3, 1>(-3.000000f, -0.250000f, -0.800000f);
  pc.col(1) = Eigen::Matrix<FloatType, 3, 1>(-3.000000f, 0.250000f, -0.800000f);
  pc.col(2) = Eigen::Matrix<FloatType, 3, 1>(-3.125000f, 0.250000f, -0.750000);
  pc.col(3) = Eigen::Matrix<FloatType, 3, 1>(-3.125000f, -0.250000f, -0.750000);
  auto hull = qh.getConvexHull(pc, true);
  assert(hull.indices().size() == 12);
  // assert(hull.vertices().size() == 4);
}

static void testHalfEdgeOutput() {
  QuickHull<FloatType> qh;

  // 8 corner vertices of a cube + tons of vertices inside.
  // Output should be a half edge mesh with 12 faces (6 cube faces with 2
  // triangles per face) and 36 half edges (3 half edges per face).
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 1008);
  // between -1 , 1
  for (int h = 1000; h < 1008; h++) {
    pc.col(h) = Eigen::Matrix<FloatType, 3, 1>(h & 1 ? -2 : 2, h & 2 ? -2 : 2,
                                               h & 4 ? -2 : 2);
  }

  HalfEdgeMesh<FloatType, size_t> mesh =
      qh.getConvexHullAsMesh(pc.data(), pc.cols(), true);
  assert(mesh.m_faces.size() == 12);
  assert(mesh.m_halfEdges.size() == 36);
  // assert(mesh.m_vertices.size() == 8);

  // Verify that for each face f, f.halfedgeIndex equals
  // next(next(next(f.halfedgeIndex))).
  for (const auto &f : mesh.m_faces) {
    size_t next = mesh.m_halfEdges[f.m_halfEdgeIndex].m_next;
    next = mesh.m_halfEdges[next].m_next;
    next = mesh.m_halfEdges[next].m_next;
    assert(next == f.m_halfEdgeIndex);
  }
}

static void testPlanes() {
  Eigen::Matrix<FloatType, 3, 1> N(1, 0, 0);
  Eigen::Matrix<FloatType, 3, 1> p(2, 0, 0);
  Plane<FloatType> P(N, p);
  auto dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(3, 0, 0), P);
  assertSameValue(dist, 1);
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(1, 0, 0), P);
  assertSameValue(dist, -1);
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(1, 0, 0), P);
  assertSameValue(dist, -1);
  N = Eigen::Matrix<FloatType, 3, 1>(2, 0, 0);
  P = Plane<FloatType>(N, p);
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(6, 0, 0), P);
  assertSameValue(dist, 8);
}
/*
static void testVertexBufferAddress() {
  QuickHull<FloatType> qh;
  std::vector<vec3> pc;
  pc.emplace_back(0, 0, 0);
  pc.emplace_back(1, 0, 0);
  pc.emplace_back(0, 1, 0);

  for (size_t i = 0; i < 2; i++) {
    const bool useOriginalIndices = i != 0;
    const auto hull = qh.getConvexHull(pc, false, useOriginalIndices);
    const auto vertices = hull.vertices();
    assert(vertices.size() > 0);
    assert((vertices.col(0) == pc[0]) == useOriginalIndices);
  }
}
*/

static void testNormals() {
  QuickHull<FloatType> qh;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc(3, 3);
  pc.col(0) = Eigen::Matrix<FloatType, 3, 1>(0, 0, 0);
  pc.col(1) = Eigen::Matrix<FloatType, 3, 1>(1, 0, 0);
  pc.col(2) = Eigen::Matrix<FloatType, 3, 1>(0, 1, 0);

  std::array<vec3, 2> normal;
  for (size_t i = 0; i < 2; i++) {
    const bool CCW = i;
    const auto hull = qh.getConvexHull(pc, CCW, false);
    const auto vertices = hull.vertices();
    const auto indices = hull.indices();
    // assert(vertices.size() == 3);
    assert(indices.size() >= 6);
    const vec3 triangle[3] = {vertices.col(indices[0]),
                              vertices.col(indices[1]),
                              vertices.col(indices[2])};
    normal[i] =
        mathutils::getTriangleNormal(triangle[0], triangle[1], triangle[2]);
  }
  const auto dot = normal[0].dot(normal[1]);
  assertSameValue(dot, -1);
}

int run() {
  // Seed RNG using Unix time
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto seed =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();

  // Setup test env
  const size_t N = 200;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 200);
  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;

  // Test 1 : Put N points inside unit cube. Result mesh must have exactly 8
  // vertices because the convex hull is the unit cube.
  for (int i = 0; i < 8; i++) {
    pc.col(i) = Eigen::Matrix<FloatType, 3, 1>(i & 1 ? -1 : 1, i & 2 ? -1 : 1,
                                               i & 4 ? -1 : 1);
  }
  hull = qh.getConvexHull(pc, true);
  assert(hull.indices().size() ==
         3 * 2 *
             6); // 6 cube faces, 2 triangles per face, 3 indices per triangle
  // true if we reduced the vertices
  // assert(hull.vertices().cols() == 8);
  auto hull2 = hull;
  assert(hull2.vertices().cols() == hull.vertices().cols());
  assert(hull2.vertices()(0, 0) == hull.vertices()(0, 0));
  assert(hull2.indices().size() == hull.indices().size());
  auto hull3 = std::move(hull);
  assert(hull.indices().size() == 0);

  // Test 1.1 : Same test, but using the original indices.
  hull = qh.getConvexHull(pc, true);
  assert(hull.indices().size() == 3 * 2 * 6);
  assert(hull.vertices().cols() == pc.cols());
  // assert((hull.vertices().col(0)) == (pc[0]));

  // Test 2 : random N points from the boundary of unit sphere. Result mesh must
  // have exactly N points.
  pc = createSphere<FloatType>(1, 50);
  hull = qh.getConvexHull(pc, true);
  assert(pc.size() == hull.vertices().size());
  hull = qh.getConvexHull(pc, true);
  // Add every vertex twice. This should not affect final mesh
  auto hull_double = qh.getConvexHull(pc.replicate(1, 2), true);
  assert(hull_double.indices().size() == hull.indices().size());

  // Test 2.1 : Multiply x components of the unit sphere vectors by a huge
  // number => essentially we get a line
  const FloatType mul = 2 * 2 * 2;
  while (true) {
    pc.row(0).array() *= mul;
    hull = qh.getConvexHull(pc, true);
    if (hull.indices().size() == 12) {
      break;
    }
  }

  // Test 3: 0D
  pc = Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 101);
  pc.array() *= 0.000001f;
  pc.col(0).setConstant(2.0f);
  hull = qh.getConvexHull(pc, true);
  assert(hull.indices().size() == 12);

  // Test 4: 2d degenerate case
  testPlanarCase();

  pc = Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Zero(3, N);
  // Test 5: first a planar circle, then make a cylinder out of it
  for (size_t i = 0; i < N; i++) {
    const FloatType alpha = (FloatType)i / N * 2 * 3.14159f;
    pc.col(i) =
        Eigen::Matrix<FloatType, 3, 1>(std::cos(alpha), 0, std::sin(alpha));
  }
  hull = qh.getConvexHull(pc, true);

  assert(hull.vertices().size() == pc.size());
  pc = pc.replicate(1, 2);
  pc.block(1, N, 1, N).array() += 1.0f;
  hull = qh.getConvexHull(pc, true);
  assert(hull.vertices().size() == pc.size());
  hull.writeWaveformOBJ("test.obj");
  std::cout << "num i = " << hull.indices().size() / 3 << '\n';
  assert(hull.indices().size() / 3 == pc.cols() * 2 - 4);

  // Test 6
  for (int x = 0;; x++) {
    pc = Eigen::Matrix<FloatType, 3, Eigen::Dynamic>(3, N);
    const FloatType l = 1;
    const FloatType r = l / (std::pow(10, x));
    for (size_t i = 0; i < N; i++) {
      vec3 p = vec3(1, 0, 0) * i * l / (N - 1);
      FloatType a = rnd(0, 2 * 3.1415f);
      vec3 d = vec3(0, std::sin(a), std::cos(a)) * r;
      pc.col(i) = p + d;
    }
    hull = qh.getConvexHull(pc, true);
    if (hull.indices().size() == 12) {
      break;
    }
  }

  // Other tests
  // testVertexBufferAddress();
  testNormals();
  testPlanes();
  testVector3();
  testHalfEdgeOutput();
  sphereTest();
  std::cout << "QuickHull tests succesfully passed." << std::endl;
  return 0;
}

} // namespace tests
} // namespace quickhull
