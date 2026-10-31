// Minimal stand-in for boost::hash_combine (same mixing formula as Boost).
#pragma once
#include <functional>
#include <cstddef>
namespace boost {
template <class T> inline void hash_combine(std::size_t& seed, const T& v) {
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}
