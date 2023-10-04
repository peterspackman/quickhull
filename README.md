In keeping with the spirit of the original: this implementation is 100% Public Domain.

Feel free to use it.

C++17 is needed to compile it.

Basic usage:

```cpp
#include "quickhull/quickhull.h"

using namespace quickhull;
QuickHull<float> qh; // Could be double as well
Eigen::Matrix3Xf pointCloud;
// Add points to point cloud
...
auto hull = qh.getConvexHull(pointCloud, true);
const auto& indices = hull.indices();
const auto& vertices = hull.vertices();
// Do what you want with the convex triangle mesh
// OR if you want only the unique points:
auto reduced_hull = hull.reduced();
const auto& indices_reduced = hull.indices();
const auto& vertices_reduced = hull.vertices();
```

The boolean parameter of getConvexHull specifies whether the resulting mesh should have its triangles in CCW orientation.

This implementation is fast, because the convex hull is internally built using a half edge mesh representation which provides quick access to adjacent faces. It is also possible to get the output convex hull as a half edge mesh:

	auto mesh = qh.getConvexHullAsMesh(&pointCloud[0].x, pointCloud.size(), true);
