#pragma once
#ifndef ANDRES_GRAPH_ADJACENCY_HXX
#define ANDRES_GRAPH_ADJACENCY_HXX

namespace andres {
namespace graph {

template<class T = std::size_t>
class Adjacency {
public:
    typedef T Value;

    Adjacency(const Value = T(), const Value = T());
    Value vertex() const;
    Value& vertex();
    Value edge() const;
    Value& edge();
    bool operator<(const Adjacency<Value>&) const;
    bool operator<=(const Adjacency<Value>&) const;
    bool operator>(const Adjacency<Value>&) const;
    bool operator>=(const Adjacency<Value>&) const;
    bool operator==(const Adjacency<Value>&) const;
    bool operator!=(const Adjacency<Value>&) const;

private:
    Value vertex_;
    Value edge_;
};

template<class T>
inline
Adjacency<T>::Adjacency(
    const Value vertex,
    const Value edge
)
:   vertex_(vertex),
    edge_(edge)
{}

template<class T>
inline typename Adjacency<T>::Value
Adjacency<T>::vertex() const {
    return vertex_;
}

template<class T>
inline typename Adjacency<T>::Value&
Adjacency<T>::vertex() {
    return vertex_;
}

template<class T>
inline typename Adjacency<T>::Value
Adjacency<T>::edge() const {
    return edge_;
}

template<class T>
inline typename Adjacency<T>::Value&
Adjacency<T>::edge() {
    return edge_;
}

template<class T>
inline bool
Adjacency<T>::operator<(
    const Adjacency<T>& in
) const {
    if(vertex_ < in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ < in.edge_;
    }
    else {
        return false;
    }
}

template<class T>
inline bool
Adjacency<T>::operator<=(
    const Adjacency<T>& in
) const {
    if(vertex_ < in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ <= in.edge_;
    }
    else {
        return false;
    }
}

template<class T>
inline bool
Adjacency<T>::operator>(
    const Adjacency<T>& in
) const {
    if(vertex_ > in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ > in.edge_;
    }
    else {
        return false;
    }
}

template<class T>
inline bool
Adjacency<T>::operator>=(
    const Adjacency<T>& in
) const {
    if(vertex_ > in.vertex_) {
        return true;
    }
    else if(vertex_ == in.vertex_) {
        return edge_ >= in.edge_;
    }
    else {
        return false;
    }
}

template<class T>
inline bool
Adjacency<T>::operator==(
    const Adjacency<T>& in
) const {
    return vertex_ == in.vertex_ && edge_ == in.edge_;
}

template<class T>
inline bool
Adjacency<T>::operator!=(
    const Adjacency<T>& in
) const {
    return !(*this == in);
}

}
}

#endif
