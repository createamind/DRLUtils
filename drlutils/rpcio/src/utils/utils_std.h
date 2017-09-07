//
// Created by Robin Huang on 12/2/16.
//

#ifndef APPC_UTILS_STD_H
#define APPC_UTILS_STD_H

#include <string>
#include <map>
#include <vector>
#include <sstream>

const std::string print_map_string(const std::map<std::string, std::string> & v);
const std::string print_vector_string(const std::vector<std::string> & v);
template<typename T>
const std::string print_vector_(const std::vector<T> & v)
{
    std::stringstream strm;
    strm << "[";
    for(auto it = v.begin(); it != v.end(); it++)
    {
        strm << *it << ",";
    }
    strm << "]";
    return strm.str();
}

const std::string print_vector_vector_string(const std::vector<std::vector<std::string>> & v);

extern const float floatNAN;

template <typename T>
T nclip(const T& n, const T& lower, const T& upper) {
    assert(lower <= upper);
    return std::max(lower, std::min(n, upper));
}

#endif //APPC_UTILS_STD_H
