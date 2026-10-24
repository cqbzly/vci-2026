#include <unordered_map>

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"

namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        Engine::SurfaceMesh curr_mesh = input;
        // We do subdivison iteratively.
        for (std::uint32_t it = 0; it < numIterations; ++it) {
            // During each iteration, we first move curr_mesh into prev_mesh.
            Engine::SurfaceMesh prev_mesh;
            prev_mesh.Swap(curr_mesh);
            // Then we create doubly connected edge list.
            DCEL G(prev_mesh);
            if (! G.IsManifold()) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Non-manifold mesh.");
                return;
            }
            // Note that here curr_mesh has already been empty.
            // We reserve memory first for efficiency.
            curr_mesh.Positions.reserve(prev_mesh.Positions.size() * 3 / 2);
            curr_mesh.Indices.reserve(prev_mesh.Indices.size() * 4);
            // Then we iteratively update currently existing vertices.
            for (std::size_t i = 0; i < prev_mesh.Positions.size(); ++i) {
                // Update the currently existing vetex v from prev_mesh.Positions.
                // Then add the updated vertex into curr_mesh.Positions.
                auto v           = G.Vertex(i);
                auto neighbors   = v->Neighbors();
                int  n = neighbors.size();
                float u = (n == 3) ? 3.0f / 16.0f : 3.0f / (8.0f * n);
                glm::vec3 neighborSum(0.0f);
                glm::vec3 oldPosition = prev_mesh.Positions[i];
                for(auto x : neighbors) {
                    neighborSum += prev_mesh.Positions[x];
                } 
                glm::vec3 newPosition = (1.0f - n * u) * oldPosition + u * neighborSum;
                curr_mesh.Positions.push_back(newPosition);
            }
            // We create an array to store indices of the newly generated vertices.
            // Note: newIndices[i][j] is the index of vertex generated on the "opposite edge" of j-th
            //       vertex in the i-th triangle.
            std::vector<std::array<std::uint32_t, 3U>> newIndices(prev_mesh.Indices.size() / 3, { ~0U, ~0U, ~0U });
            // Iteratively process each halfedge.
            for (auto e : G.Edges()) {
                // newIndices[face index][vertex index] = index of the newly generated vertex
                newIndices[G.IndexOf(e->Face())][e->EdgeLabel()] = curr_mesh.Positions.size();
                auto eTwin                                       = e->TwinEdgeOr(nullptr);
                glm::vec3 fromPosition = prev_mesh.Positions[e->From()];
                glm::vec3 toPosition = prev_mesh.Positions[e->To()];
                glm::vec3 curPosition;
                // eTwin stores the twin halfedge.
                if (! eTwin) {
                    // When there is no twin halfedge (so, e is a boundary edge):
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    curPosition = (fromPosition + toPosition) / 2.0f;
                    curr_mesh.Positions.push_back(curPosition);
                } else {
                    // When the twin halfedge exists, we should also record:
                    //     newIndices[face index][vertex index] = index of the newly generated vertex
                    // Because G.Edges() will only traverse once for two halfedges,
                    //     we have to record twice.
                    newIndices[G.IndexOf(eTwin->Face())][e->TwinEdge()->EdgeLabel()] = curr_mesh.Positions.size();
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    auto oppositePosition0 = prev_mesh.Positions[e->OppositeVertex()];
                    auto oppositePosition1 = prev_mesh.Positions[eTwin->OppositeVertex()];
                    curPosition = (3.0f / 8.0f) * (fromPosition + toPosition) + (1.0f / 8.0f) * (oppositePosition0 + oppositePosition1);
                    curr_mesh.Positions.push_back(curPosition);
                }
            }

            // Here we've already build all the vertices.
            // Next, it's time to reconstruct face indices.
            for (std::size_t i = 0; i < prev_mesh.Indices.size(); i += 3U) {
                // For each face F in prev_mesh, we should create 4 sub-faces.
                // v0,v1,v2 are indices of vertices in F.
                // m0,m1,m2 are generated vertices on the edges of F.
                auto v0           = prev_mesh.Indices[i + 0U];
                auto v1           = prev_mesh.Indices[i + 1U];
                auto v2           = prev_mesh.Indices[i + 2U];
                auto [m0, m1, m2] = newIndices[i / 3U];
                // Note: m0 is on the opposite edge (v1-v2) to v0.
                // Please keep the correct indices order (consistent with order v0-v1-v2)
                //     when inserting new face indices.
                // toInsert[i][j] stores the j-th vertex index of the i-th sub-face.
                std::uint32_t toInsert[4][3] = {
                    // your code here:
                    {v0, m2, m1},
                    {v1, m0, m2},
                    {v2, m1, m0},
                    {m0, m1, m2}
                };
                // Do insertion.
                curr_mesh.Indices.insert(
                    curr_mesh.Indices.end(),
                    reinterpret_cast<std::uint32_t *>(toInsert),
                    reinterpret_cast<std::uint32_t *>(toInsert) + 12U
                );
            }

            if (curr_mesh.Positions.size() == 0) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Empty mesh.");
                output = input;
                return;
            }
        }
        // Update output.
        output.Swap(curr_mesh);
    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // Copy.
        output = input;
        // Reset output.TexCoords.
        output.TexCoords.resize(input.Positions.size(), glm::vec2 { 0 });

        // Build DCEL.
        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::Parameterization(..): non-manifold mesh.");
            return;
        }

        // Set boundary UVs for boundary vertices.
        // your code here: directly edit output.TexCoords

        std::uint32_t start = ~0U;
        for(std::uint32_t i = 0; i < input.Positions.size(); i++) {
            auto vertex = G.Vertex(i);
            if(vertex != nullptr && vertex->OnBoundary()) {
                start = i;
                break;
            }
        }
        std::vector<std::uint32_t> boundary;
        std::uint32_t cur = start;
        std::uint32_t pre = ~0U;
        do {
            boundary.push_back(cur);
            auto [neighbor0, neighbor1] = G.Vertex(cur)->BoundaryNeighbors();
            std::uint32_t nxt = neighbor0 == pre ? neighbor1 : neighbor0;
            pre = cur;
            cur = nxt;
        } while(cur != start);
        const float pi = glm::pi<float>();
        const std::uint32_t cnt = boundary.size();
        for(std::size_t i = 0; i < cnt; i ++) {
            float angle = 2.0f * pi * i / cnt;
            output.TexCoords[boundary[i]] = glm::vec2(0.5f + 0.5f * std::cos(angle), 0.5f + 0.5f * std::sin(angle));
        }
        // Solve equation via Gauss-Seidel Iterative Method.
        for (int k = 0; k < numIterations; ++k) {
            // your code here:
            for(std::size_t i = 0; i < input.Positions.size(); i++) {
                auto vertex = G.Vertex(i);
                if(vertex->OnBoundary()) continue;
                auto neighbors = vertex->Neighbors();
                glm::vec2 sum(0.0f);
                for(auto neighbor : neighbors) {
                    sum +=output.TexCoords[neighbor];
                }
                output.TexCoords[i] = sum / static_cast<float>(neighbors.size());
            }

        }
    }

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float simplification_ratio) {

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-watertight mesh.");
            return;
        }

        // Copy.
        output = input;

        // Compute Kp matrix of the face f.
        auto UpdateQ {
            [&G, &output] (DCEL::Triangle const * f) -> glm::mat4 {
                glm::mat4 Kp;
                // your code here:
                glm::vec3 p0 = output.Positions[f->VertexIndex(0)];
                glm::vec3 p1 = output.Positions[f->VertexIndex(1)];
                glm::vec3 p2 = output.Positions[f->VertexIndex(2)];
                glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
                float len = glm::length(n);
                if(len <= 1e-12f) return glm::mat4(0);
                n /= len;
                float d = -glm::dot(p0, n);
                glm::vec4 p {n, d};
                Kp = glm::outerProduct(p, p);
                return Kp;
            }
        };

        // The struct to record contraction info.
        struct ContractionPair {
            DCEL::HalfEdge const * edge;            // which edge to contract; if $edge == nullptr$, it means this pair is no longer valid
            glm::vec4              targetPosition;  // the targetPosition $v$ for vertex $edge->From()$ to move to
            float                  cost;            // the cost $v.T * Qbar * v$
        };

        // Given an edge (v1->v2), the positions of its two endpoints (p1, p2) and the Q matrix (Q1+Q2),
        //     return the ContractionPair struct.
        static constexpr auto MakePair {
            [] (DCEL::HalfEdge const * edge,
                glm::vec3 const & p1,
                glm::vec3 const & p2,
                glm::mat4 const & Q
            ) -> ContractionPair {
                // your code here:
                glm::mat3 A(Q);
                glm::vec3 b(Q[3][0],Q[3][1],Q[3][2]);
                ContractionPair result;
                result.edge = edge;
                auto costAt {
                    [&Q] (glm::vec3 const &x) {
                        glm::vec4 v(x, 1.0f);
                        return glm::dot(v, Q * v);
                    }
                };
                if(std::abs(glm::determinant(A)) < 1e-3f) {
                    glm::vec3 candidates[3] = {p1, p2, (p1 + p2) * 0.5f};
                    for(int k = 0; k < 3; k++) {
                        float cost = costAt(candidates[k]);
                        if(k == 0 || cost < result.cost) {
                            result.targetPosition = {candidates[k], 1.0f};
                            result.cost = cost;
                        }
                    }
                }
                else {
                    glm::vec3 v = -glm::inverse(A) * b;
                    result.targetPosition = {v, 1.0f};
                    result.cost = costAt(v);
                }
                return result;
            }
        };

        // pair_map: map EdgeIdx to index of $pairs$
        // pairs:    store ContractionPair
        // Qv:       $Qv[idx]$ is the Q matrix of vertex with index $idx$
        // Kf:       $Kf[idx]$ is the Kp matrix of face with index $idx$
        std::unordered_map<DCEL::EdgeIdx, std::size_t> pair_map; 
        std::vector<ContractionPair>                  pairs; 
        std::vector<glm::mat4>                         Qv(G.NumOfVertices(), glm::mat4(0));
        std::vector<glm::mat4>                         Kf(G.NumOfFaces(),    glm::mat4(0));

        // Initially, we compute Q matrix for each faces and it accumulates at each vertex.
        for (auto f : G.Faces()) {
            auto Q                 = UpdateQ(f);
            Qv[f->VertexIndex(0)] += Q;
            Qv[f->VertexIndex(1)] += Q;
            Qv[f->VertexIndex(2)] += Q;
            Kf[G.IndexOf(f)]       = Q;
        }

        pair_map.reserve(G.NumOfFaces() * 3);
        pairs.reserve(G.NumOfFaces() * 3 / 2);

        // Initially, we make pairs from all the contractable edges.
        for (auto e : G.Edges()) {
            if (! G.IsContractable(e)) continue;
            auto v1                            = e->From();
            auto v2                            = e->To();
            auto pair                          = MakePair(e, input.Positions[v1], input.Positions[v2], Qv[v1] + Qv[v2]);
            pair_map[G.IndexOf(e)]             = pairs.size();
            pair_map[G.IndexOf(e->TwinEdge())] = pairs.size();
            pairs.emplace_back(pair);
        }

        // Loop until the number of vertices is less than $simplification_ratio * initial_size$.
        while (G.NumOfVertices() > simplification_ratio * Qv.size()) {
            // Find the contractable pair with minimal cost.
            std::size_t min_idx = ~0;
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                if (! pairs[i].edge) continue;
                if (!~min_idx || pairs[i].cost < pairs[min_idx].cost) {
                    if (G.IsContractable(pairs[i].edge)) min_idx = i;
                    else pairs[i].edge = nullptr;
                }
            }
            if (!~min_idx) break;

            // top:    the contractable pair with minimal cost
            // v1:     the reserved vertex
            // v2:     the removed vertex
            // result: the contract result
            // ring:   the edge ring of vertex v1
            ContractionPair & top    = pairs[min_idx];
            auto               v1     = top.edge->From();
            auto               v2     = top.edge->To();
            auto               result = G.Contract(top.edge);
            auto               ring   = G.Vertex(v1)->Ring();

            top.edge             = nullptr;            // The contraction has already been done, so the pair is no longer valid. Mark it as invalid.
            output.Positions[v1] = top.targetPosition; // Update the positions.

            // We do something to repair $pair_map$ and $pairs$ because some edges and vertices no longer exist.
            for (int i = 0; i < 2; ++i) {
                DCEL::EdgeIdx removed           = G.IndexOf(result.removed_edges[i].first);
                DCEL::EdgeIdx collapsed         = G.IndexOf(result.collapsed_edges[i].second);
                pairs[pair_map[removed]].edge   = result.collapsed_edges[i].first;
                pairs[pair_map[collapsed]].edge = nullptr;
                pair_map[collapsed]             = pair_map[G.IndexOf(result.collapsed_edges[i].first)];
            }

            // For the two wing vertices, each of them lose one incident face.
            // So, we update the Q matrix.
            Qv[result.removed_faces[0].first] -= Kf[G.IndexOf(result.removed_faces[0].second)];
            Qv[result.removed_faces[1].first] -= Kf[G.IndexOf(result.removed_faces[1].second)];

            // For the vertex v1, Q matrix should be recomputed.
            // And as the position of v1 changed, all the vertices which are on the ring of v1 should update their Q matrix as well.
            Qv[v1] = glm::mat4(0);
            for (auto e : ring) {
                // your code here:
                //     1. Compute the new Kp matrix for $e->Face()$.
                //     2. According to the difference between the old Kp (in $Kf$) and the new Kp (computed in step 1),
                //        update Q matrix of each vertex on the ring (update $Qv$).
                //     3. Update Q matrix of vertex v1 as well (update $Qv$).
                //     4. Update $Kf$.
                auto f = e->Face();
                auto kp = UpdateQ(e->Face());
                auto diff = kp - Kf[G.IndexOf(f)];
                Qv[e->From()] += diff;
                Qv[e->To()] += diff;
                Qv[v1] += kp;
                Kf[G.IndexOf(f)] = kp;
            }

            // Finally, as the Q matrix changed, we should update the relative $ContractionPair$ in $pairs$.
            // Any pair with the Q matrix of its endpoints changed, should be remade by $MakePair$.
            // your code here:
            auto remake = [&](DCEL::HalfEdge const *h) {
                auto it = pair_map.find(G.IndexOf(h));
                if(it == pair_map.end()) return;
                auto a = h->From();
                auto b = h->To();
                pairs[it->second] = MakePair(h, output.Positions[a], output.Positions[b], Qv[a] + Qv[b]);
            };
            for(auto e :ring) {
                for(auto g : G.Vertex(e->From())->Ring()) {
                    remake(g->PrevEdge());
                }
            }
        }

        // In the end, we check if the result mesh is watertight and manifold.
        if (! G.DebugWatertightManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Result is not watertight manifold.");
        }
        

        auto exported = G.ExportMesh();
        output.Indices.swap(exported.Indices);
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // Define function to compute cotangent value of the angle v1-vAngle-v2
        static constexpr auto GetCotangent {
            [] (glm::vec3 vAngle, glm::vec3 v1, glm::vec3 v2) -> float {
                // your code here:
                glm::vec3 a = v1 - vAngle;
                glm::vec3 b = v2 - vAngle;
                float cross_len = glm::length(glm::cross(a, b));
                if (cross_len < 1e-12f) return 0.0f;
                return glm::dot(a, b) / cross_len;
            }
        };

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-watertight mesh.");
            return;
        }

        Engine::SurfaceMesh prev_mesh;
        prev_mesh.Positions = input.Positions;
        for (std::uint32_t iter = 0; iter < numIterations; ++iter) {
            Engine::SurfaceMesh curr_mesh = prev_mesh;
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                // your code here: curr_mesh.Positions[i] = ...
                auto ring = G.Vertex(i)->Ring();
                glm::vec3 sum(0.0f);
                float wsum = 0.0f;
                for(auto e : ring) {
                    auto *h = e->PrevEdge();
                    auto j = h->To();
                    float w;
                    if(useUniformWeight) {
                        w = 1.0f;
                    }
                    else {
                        auto c1 = h->OppositeVertex();
                        auto c2 = h->TwinOppositeVertex();
                        auto p1 = prev_mesh.Positions[c1];
                        auto p2 = prev_mesh.Positions[c2];
                        auto p3 = prev_mesh.Positions[i];
                        auto p4 = prev_mesh.Positions[j];
                        w = GetCotangent(p1, p3, p4) + GetCotangent(p2, p3, p4);
                        if(w < 0.0f) w = 0.0f;
                    }
                    sum += w * prev_mesh.Positions[j];
                    wsum += w;
                }
                if(wsum > 1e-12f) curr_mesh.Positions[i] = (1.0f - lambda) * prev_mesh.Positions[i] + lambda * (sum / wsum);
            }
            // Move curr_mesh to prev_mesh.
            prev_mesh.Swap(curr_mesh);
        }
        // Move prev_mesh to output.
        output.Swap(prev_mesh);
        // Copy indices from input.
        output.Indices = input.Indices;
    }

    /******************* 5. Marching Cubes *****************/
    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here:
        output.Positions.clear();
        output.Indices.clear();
        std::unordered_map<std::uint64_t, std::uint32_t> vertex;
        vertex.reserve(n * n * 4);
        static constexpr int kEdgeCorner[12][2] = {
            { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 },   // 沿 x
            { 0, 2 }, { 4, 6 }, { 1, 3 }, { 5, 7 },   // 沿 y
            { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },   // 沿 z
        };
        auto makeKey = [](int axis, glm::ivec3 const &s) -> std::uint64_t {
            return ((uint64_t)axis << 48) | ((uint64_t)s.x << 32) | ((uint64_t)s.y << 16) | s.z;
        };
        for(int z = 0; z < n; z++) {
            for(int y = 0; y < n; y++) {
                for(int x = 0; x < n; x++) {
                    glm::ivec3 base(x, y, z);
                    glm::vec3 p[8];
                    float v[8];
                    uint32_t cubeIndex = 0;
                    for(int i = 0; i < 8; i ++) {
                        p[i] = grid_min + dx * glm::vec3(x + (i & 1), y + (i >> 1 & 1), z + (i >> 2));
                        v[i] = sdf(p[i]);
                        if(v[i] > 0.0f) cubeIndex |= (1 << i);
                    }
                    std::uint32_t edgeState = c_EdgeStateTable[cubeIndex];
                    if(edgeState == 0) continue;
                    std::uint32_t edgeVertex[12] = {};
                    for(int j = 0;j < 12; j++){
                        if(!(edgeState >> j & 1)) continue;
                        int a = kEdgeCorner[j][0];
                        int b = kEdgeCorner[j][1];
                        float t = v[a] / (v[a] - v[b]);
                        glm::vec3 pos = p[a] + t * (p[b] - p[a]);
                        int axis = j >> 2;
                        glm::ivec3 off(0);
                        if(j & 1) off[(axis + 1) % 3] = 1;
                        if(j >> 1 & 1) off[(axis + 2) % 3] = 1;
                        std::uint64_t key = makeKey(axis, base + off);
                        auto [it, inserted] = vertex.try_emplace(key, output.Positions.size());
                        if(inserted) output.Positions.push_back(pos);
                        edgeVertex[j] = it->second;
                    }
                    for(int k = 0; c_EdgeOrdsTable[cubeIndex][k] != -1; k += 3) {
                        output.Indices.push_back(edgeVertex[c_EdgeOrdsTable[cubeIndex][k]]);
                        output.Indices.push_back(edgeVertex[c_EdgeOrdsTable[cubeIndex][k + 2]]);
                        output.Indices.push_back(edgeVertex[c_EdgeOrdsTable[cubeIndex][k + 1]]);
                    }
                }
            }
        }
    }
} // namespace VCX::Labs::GeometryProcessing
