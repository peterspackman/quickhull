#pragma once
#include "mesh.hpp"
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace quickhull {

template <typename T> class ConvexHull {
  Eigen::Matrix<T, 3, Eigen::Dynamic> m_vertices;
  std::vector<size_t> m_indices;

public:
  ConvexHull() {}

  // Copy constructor
  ConvexHull(const ConvexHull &o) {
    m_indices = o.m_indices;
    m_vertices = o.m_vertices;
  }

  ConvexHull &operator=(const ConvexHull &o) {
    if (&o == this) {
      return *this;
    }
    m_indices = o.m_indices;
    m_vertices = o.m_vertices;
    return *this;
  }

  ConvexHull(ConvexHull &&o) {
    m_indices = std::move(o.m_indices);
    m_vertices = std::move(o.m_vertices);
  }

  ConvexHull &operator=(ConvexHull &&o) {
    if (&o == this) {
      return *this;
    }
    m_indices = std::move(o.m_indices);
    m_vertices = std::move(o.m_vertices);
    return *this;
  }

  // Construct vertex and index buffers from half edge mesh and pointcloud
  ConvexHull(const MeshBuilder<T> &mesh,
             const Eigen::Matrix<T, 3, Eigen::Dynamic> &pointCloud, bool CCW) {

    std::vector<bool> faceProcessed(mesh.m_faces.size(), false);
    std::vector<size_t> faceStack;
    std::unordered_map<size_t, size_t>
        vertexIndexMapping; // Map vertex indices from original point cloud to
                            // the new mesh vertex indices
    for (size_t i = 0; i < mesh.m_faces.size(); i++) {
      if (!mesh.m_faces[i].isDisabled()) {
        faceStack.push_back(i);
        break;
      }
    }
    if (faceStack.size() == 0) {
      return;
    }

    const size_t iCCW = CCW ? 1 : 0;
    const size_t finalMeshFaceCount =
        mesh.m_faces.size() - mesh.m_disabledFaces.size();
    m_indices.reserve(finalMeshFaceCount * 3);

    while (faceStack.size()) {
      auto it = faceStack.end() - 1;
      size_t top = *it;
      assert(!mesh.m_faces[top].isDisabled());
      faceStack.erase(it);
      if (faceProcessed[top]) {
        continue;
      } else {
        faceProcessed[top] = true;
        auto halfEdges = mesh.getHalfEdgeIndicesOfFace(mesh.m_faces[top]);
        size_t adjacent[] = {
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[0]].m_opp].m_face,
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[1]].m_opp].m_face,
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[2]].m_opp].m_face};
        for (auto a : adjacent) {
          if (!faceProcessed[a] && !mesh.m_faces[a].isDisabled()) {
            faceStack.push_back(a);
          }
        }
        auto vertices = mesh.getVertexIndicesOfFace(mesh.m_faces[top]);
        m_indices.push_back(vertices[0]);
        m_indices.push_back(vertices[1 + iCCW]);
        m_indices.push_back(vertices[2 - iCCW]);
      }
    }
    m_vertices = pointCloud;
  }

  inline const auto &indices() const { return m_indices; }
  inline auto &indices() { return m_indices; }
  inline const auto &vertices() const { return m_vertices; }
  inline auto &vertices() { return m_vertices; }

  // Export the mesh to a Waveform OBJ file
  void writeWaveformOBJ(const std::string &filename,
                        const std::string &objectName = "quickhull") const {
    std::ofstream objFile;
    objFile.open(filename);
    objFile << "o " << objectName << "\n";
    for (int i = 0; i < m_vertices.cols(); i++) {
      objFile << "v " << m_vertices(0, i) << " " << m_vertices(1, i) << " "
              << m_vertices(2, i) << "\n";
    }
    size_t triangleCount = m_indices.size() / 3;
    for (size_t i = 0; i < triangleCount; i++) {
      objFile << "f " << m_indices[i * 3] + 1 << " " << m_indices[i * 3 + 1] + 1
              << " " << m_indices[i * 3 + 2] + 1 << "\n";
    }
    objFile.close();
  }
};

} // namespace quickhull
