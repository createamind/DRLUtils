//
// Created by Robin Huang on 12/2/16.
//

#include "utils_std.h"
#include <sstream>
#include <limits>

const float floatNAN = std::numeric_limits<float>::quiet_NaN(); //std::nanf("NAN");

const std::string print_map_string(const std::map<std::string, std::string> & v)
{
    std::stringstream strm;
    strm << "[";
    for(auto it = v.begin(); it != v.end(); it++)
    {
        strm << it->first << "=" << it->second << ",";
    }
    strm << "]";
    return strm.str();
}

const std::string print_vector_string(const std::vector<std::string> & v)
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

const std::string print_vector_vector_string(const std::vector<std::vector<std::string>> & v)
{
    std::stringstream strm;
    strm << "[";
    for(auto it = v.begin(); it != v.end(); it++)
    {
        strm << "[";
        for(auto _it = (*it).begin(); _it != (*it).end(); _it++)
            strm << *_it << ",";
        strm << "],";
    }
    strm << "]";
    return strm.str();
}
