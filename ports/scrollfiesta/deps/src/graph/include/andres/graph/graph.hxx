#pragma once
#ifndef ANDRES_GRAPH_HXX
#define ANDRES_GRAPH_HXX

#include <cassert>
#include <cstddef>
#include <iterator>
#include <vector>
#include <set>
#include <iostream>
#include <utility>
#include <algorithm>

#include "adjacency.hxx"
#include "subgraph.hxx"
#include "visitor.hxx"
#include "detail/graph.hxx"

namespace andres {

namespace graph {

template<typename VISITOR = IdleGraphVisitor<std::size_t> >
class Graph {
public:
    typedef VISITOR Visitor;
    typedef detail::VertexIterator VertexIterator;
    typedef detail::EdgeIterator EdgeIterator;
    typedef detail::Adjacencies::const_iterator AdjacencyIterator;
    typedef typename AdjacencyIterator::value_type AdjacencyType;

       Graph(const Visitor& = Visitor());
    Graph(const std::size_t, const Visitor& = Visitor());
    void assign(const Visitor& = Visitor());
    void assign(const std::size_t, const Visitor& = Visitor());
    void reserveVertices(const std::size_t);
    void reserveEdges(const std::size_t);

       VertexIterator verticesFromVertexBegin(const std::size_t) const;
    VertexIterator verticesFromVertexEnd(const std::size_t) const;
    VertexIterator verticesToVertexBegin(const std::size_t) const;
    VertexIterator verticesToVertexEnd(const std::size_t) const;
    EdgeIterator edgesFromVertexBegin(const std::size_t) const;
    EdgeIterator edgesFromVertexEnd(const std::size_t) const;
    EdgeIterator edgesToVertexBegin(const std::size_t) const;
    EdgeIterator edgesToVertexEnd(const std::size_t) const;
    AdjacencyIterator adjacenciesFromVertexBegin(const std::size_t) const;
    AdjacencyIterator adjacenciesFromVertexEnd(const std::size_t) const;
    AdjacencyIterator adjacenciesToVertexBegin(const std::size_t) const;
    AdjacencyIterator adjacenciesToVertexEnd(const std::size_t) const;

       std::size_t numberOfVertices() const;
    std::size_t numberOfEdges() const;
    std::size_t numberOfEdgesFromVertex(const std::size_t) const;
    std::size_t numberOfEdgesToVertex(const std::size_t) const;
    std::size_t vertexOfEdge(const std::size_t, const std::size_t) const;
    std::size_t edgeFromVertex(const std::size_t, const std::size_t) const;
    std::size_t edgeToVertex(const std::size_t, const std::size_t) const;
    std::size_t vertexFromVertex(const std::size_t, const std::size_t) const;
    std::size_t vertexToVertex(const std::size_t, const std::size_t) const;
    const AdjacencyType& adjacencyFromVertex(const std::size_t, const std::size_t) const;
    const AdjacencyType& adjacencyToVertex(const std::size_t, const std::size_t) const;
    std::pair<bool, std::size_t> findEdge(const std::size_t, const std::size_t) const;
    bool multipleEdgesEnabled() const;

       std::size_t insertVertex();
    std::size_t insertVertices(const std::size_t);
    std::size_t insertEdge(const std::size_t, const std::size_t);
    void eraseVertex(const std::size_t);
    void eraseEdge(const std::size_t);
    bool& multipleEdgesEnabled();

private:
    typedef detail::Adjacencies Vertex;
    typedef detail::Edge<false> Edge;

    void insertAdjacenciesForEdge(const std::size_t);
    void eraseAdjacenciesForEdge(const std::size_t);

    std::vector<Vertex> vertices_;
    std::vector<Edge> edges_;
    bool multipleEdgesEnabled_;
    Visitor visitor_;
};

template<typename VISITOR>
inline
Graph<VISITOR>::Graph(
    const Visitor& visitor
)
:   vertices_(),
    edges_(),
    multipleEdgesEnabled_(false),
    visitor_(visitor)
{}

template<typename VISITOR>
inline
Graph<VISITOR>::Graph(
    const std::size_t numberOfVertices,
    const Visitor& visitor
)
:   vertices_(numberOfVertices),
    edges_(),
    multipleEdgesEnabled_(false),
    visitor_(visitor)
{
    visitor_.insertVertices(0, numberOfVertices);
}

template<typename VISITOR>
inline void
Graph<VISITOR>::assign(
    const Visitor& visitor
) {
    vertices_.clear();
    edges_.clear();
    multipleEdgesEnabled_ = false;
    visitor_ = visitor;
}

template<typename VISITOR>
inline void
Graph<VISITOR>::assign(
    const std::size_t numberOfVertices,
    const Visitor& visitor
) {
    vertices_.resize(numberOfVertices);
    std::fill(vertices_.begin(), vertices_.end(), Vertex());
    edges_.clear();
    multipleEdgesEnabled_ = false;
    visitor_ = visitor;
    visitor_.insertVertices(0, numberOfVertices);
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::numberOfVertices() const {
    return vertices_.size();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::numberOfEdges() const {
    return edges_.size();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::numberOfEdgesFromVertex(
    const std::size_t vertex
) const {
    return vertices_[vertex].size();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::numberOfEdgesToVertex(
    const std::size_t vertex
) const {
    return vertices_[vertex].size();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::vertexOfEdge(
    const std::size_t edge,
    const std::size_t j
) const {
    assert(j < 2);

    return edges_[edge][j];
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::edgeFromVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j].edge();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::edgeToVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j].edge();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::vertexFromVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j].vertex();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::vertexToVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j].vertex();
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::insertVertex() {
    vertices_.push_back(Vertex());
    visitor_.insertVertex(vertices_.size() - 1);
    return vertices_.size() - 1;
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::insertVertices(
    const std::size_t number
) {
    std::size_t position = vertices_.size();
    vertices_.insert(vertices_.end(), number, Vertex());
    visitor_.insertVertices(position, number);
    return position;
}

template<typename VISITOR>
inline std::size_t
Graph<VISITOR>::insertEdge(
    const std::size_t vertexIndex0,
    const std::size_t vertexIndex1
) {
    assert(vertexIndex0 < numberOfVertices());
    assert(vertexIndex1 < numberOfVertices());

    if(multipleEdgesEnabled()) {
insertEdgeMark:
        edges_.push_back(Edge(vertexIndex0, vertexIndex1));
        std::size_t edgeIndex = edges_.size() - 1;
        insertAdjacenciesForEdge(edgeIndex);
        visitor_.insertEdge(edgeIndex);
        return edgeIndex;
    }
    else {
        std::pair<bool, std::size_t> p = findEdge(vertexIndex0, vertexIndex1);
        if(p.first) {
            return p.second;
        }
        else {
            goto insertEdgeMark;
        }
    }
}

template<typename VISITOR>
void
Graph<VISITOR>::eraseVertex(
    const std::size_t vertexIndex
) {
    assert(vertexIndex < numberOfVertices());

       while(vertices_[vertexIndex].size() != 0) {
        eraseEdge(vertices_[vertexIndex].begin()->edge());
    }

    if(vertexIndex == numberOfVertices()-1) {
        vertices_.pop_back();
        visitor_.eraseVertex(vertexIndex);
    }
    else {

               std::size_t movingVertexIndex = numberOfVertices() - 1;
        std::set<std::size_t> affectedEdgeIndices;
        for(Vertex::const_iterator it = vertices_[movingVertexIndex].begin();
        it != vertices_[movingVertexIndex].end(); ++it) {
            affectedEdgeIndices.insert(it->edge());
        }

               for(std::set<std::size_t>::const_iterator it = affectedEdgeIndices.begin();
        it != affectedEdgeIndices.end(); ++it) {
                       eraseAdjacenciesForEdge(*it);

                       for(std::size_t j=0; j<2; ++j) {
                if(edges_[*it][j] == movingVertexIndex) {
                    edges_[*it][j] = vertexIndex;
                }
            }
                       if(edges_[*it][0] > edges_[*it][1]) {
                std::swap(edges_[*it][0], edges_[*it][1]);
            }
        }

               vertices_[vertexIndex] = vertices_[movingVertexIndex];
        vertices_.pop_back();

               for(std::set<std::size_t>::const_iterator it = affectedEdgeIndices.begin();
        it != affectedEdgeIndices.end(); ++it) {
            insertAdjacenciesForEdge(*it);
        }

        visitor_.eraseVertex(vertexIndex);
        visitor_.relabelVertex(movingVertexIndex, vertexIndex);
    }
}

template<typename VISITOR>
inline void
Graph<VISITOR>::eraseEdge(
    const std::size_t edgeIndex
) {
    assert(edgeIndex < numberOfEdges());

    eraseAdjacenciesForEdge(edgeIndex);
    if(edgeIndex == numberOfEdges() - 1) {
        edges_.pop_back();
        visitor_.eraseEdge(edgeIndex);
    }
    else {
        std::size_t movingEdgeIndex = numberOfEdges() - 1;
        eraseAdjacenciesForEdge(movingEdgeIndex);
        edges_[edgeIndex] = edges_[movingEdgeIndex];
        edges_.pop_back();
        insertAdjacenciesForEdge(edgeIndex);
        visitor_.eraseEdge(edgeIndex);
        visitor_.relabelEdge(movingEdgeIndex, edgeIndex);
    }
}

template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator
Graph<VISITOR>::verticesFromVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator
Graph<VISITOR>::verticesFromVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator
Graph<VISITOR>::verticesToVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::VertexIterator
Graph<VISITOR>::verticesToVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator
Graph<VISITOR>::edgesFromVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator
Graph<VISITOR>::edgesFromVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator
Graph<VISITOR>::edgesToVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::EdgeIterator
Graph<VISITOR>::edgesToVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator
Graph<VISITOR>::adjacenciesFromVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator
Graph<VISITOR>::adjacenciesFromVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator
Graph<VISITOR>::adjacenciesToVertexBegin(
    const std::size_t vertex
) const {
    return vertices_[vertex].begin();
}

template<typename VISITOR>
inline typename Graph<VISITOR>::AdjacencyIterator
Graph<VISITOR>::adjacenciesToVertexEnd(
    const std::size_t vertex
) const {
    return vertices_[vertex].end();
}

template<typename VISITOR>
inline void
Graph<VISITOR>::reserveVertices(
    const std::size_t number
) {
    vertices_.reserve(number);
}

template<typename VISITOR>
inline void
Graph<VISITOR>::reserveEdges(
    const std::size_t number
) {
    edges_.reserve(number);
}

template<typename VISITOR>
inline const typename Graph<VISITOR>::AdjacencyType&
Graph<VISITOR>::adjacencyFromVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j];
}

template<typename VISITOR>
inline const typename Graph<VISITOR>::AdjacencyType&
Graph<VISITOR>::adjacencyToVertex(
    const std::size_t vertex,
    const std::size_t j
) const {
    return vertices_[vertex][j];
}

template<typename VISITOR>
inline std::pair<bool, std::size_t>
Graph<VISITOR>::findEdge(
    const std::size_t vertex0,
    const std::size_t vertex1
) const {
    assert(vertex0 < numberOfVertices());
    assert(vertex1 < numberOfVertices());

    std::size_t v0 = vertex0;
    std::size_t v1 = vertex1;
    if(numberOfEdgesFromVertex(vertex1) < numberOfEdgesFromVertex(vertex0)) {
        v0 = vertex1;
        v1 = vertex0;
    }
    VertexIterator it = std::lower_bound(
        verticesFromVertexBegin(v0),
        verticesFromVertexEnd(v0),
        v1
    );
    if(it != verticesFromVertexEnd(v0) && *it == v1) {
               const std::size_t j = std::distance(verticesFromVertexBegin(v0), it);
        return std::make_pair(true, edgeFromVertex(v0, j));
    }
    else {
        return std::make_pair(false, 0);
    }
}

template<typename VISITOR>
inline bool
Graph<VISITOR>::multipleEdgesEnabled() const {
    return multipleEdgesEnabled_;
}

template<typename VISITOR>
inline bool&
Graph<VISITOR>::multipleEdgesEnabled() {
    return multipleEdgesEnabled_;
}

template<typename VISITOR>
inline void
Graph<VISITOR>::insertAdjacenciesForEdge(
    const std::size_t edgeIndex
) {
    const Edge& edge = edges_[edgeIndex];
    const std::size_t vertexIndex0 = edge[0];
    const std::size_t vertexIndex1 = edge[1];
    vertices_[vertexIndex0].insert(
        AdjacencyType(vertexIndex1, edgeIndex)
    );
    if(vertexIndex1 != vertexIndex0) {
        vertices_[vertexIndex1].insert(
            AdjacencyType(vertexIndex0, edgeIndex)
        );
    }
}

template<typename VISITOR>
inline void
Graph<VISITOR>::eraseAdjacenciesForEdge(
    const std::size_t edgeIndex
) {
    const Edge& edge = edges_[edgeIndex];
    const std::size_t vertexIndex0 = edge[0];
    const std::size_t vertexIndex1 = edge[1];
    Vertex& vertex0 = vertices_[vertexIndex0];
    Vertex& vertex1 = vertices_[vertexIndex1];

    AdjacencyType adj(vertexIndex1, edgeIndex);
    RandomAccessSet<AdjacencyType>::iterator it = vertex0.find(adj);
    assert(it != vertex0.end());
    if (it != vertex0.end())
        vertex0.erase(it);

    if(vertexIndex1 != vertexIndex0) {
        adj.vertex() = vertexIndex0;
        it = vertex1.find(adj);
        assert(it != vertex1.end());
        if (it != vertex1.end())
            vertex1.erase(it);
    }
}

}
}

#endif
