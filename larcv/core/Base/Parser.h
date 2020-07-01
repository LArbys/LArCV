#ifndef __LARCVBASE_PARSER_H__
#define __LARCVBASE_PARSER_H__

#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <functional> 
#include <cctype>
#include <locale>

namespace larcv {
  namespace parser {

    // STRING WHITESPACE CLEANING FROM: https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
    // trim from start (in place)
    static inline void ltrim(std::string &s) {
      s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                      std::not1(std::ptr_fun<int, int>(std::isspace))));
    }
    
    // trim from end (in place)
    static inline void rtrim(std::string &s) {
      s.erase(std::find_if(s.rbegin(), s.rend(),
                           std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    }
    
    // trim from both ends (in place)
    static inline void trim(std::string &s) {
      ltrim(s);
      rtrim(s);
    }
    
    // trim from start (copying)
    static inline std::string ltrim_copy(std::string s) {
      ltrim(s);
    return s;
    }
    
    // trim from end (copying)
    static inline std::string rtrim_copy(std::string s) {
      rtrim(s);
      return s;
    }
    
    // trim from both ends (copying)
    static inline std::string trim_copy(std::string s) {
      trim(s);
      return s;
    }
    
    template <class T>
    T FromString( const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::string    FromString< std::string    > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> float          FromString< float          > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> double         FromString< double         > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> short          FromString< short          > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> int            FromString< int            > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> long           FromString< long           > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> unsigned short FromString< unsigned short > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> unsigned int   FromString< unsigned int   > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> unsigned long  FromString< unsigned long  > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> bool           FromString< bool           > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< std::string    > FromString< std::vector< std::string    > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< float          > FromString< std::vector< float          > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< double         > FromString< std::vector< double         > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< short          > FromString< std::vector< short          > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< int            > FromString< std::vector< int            > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< long           > FromString< std::vector< long           > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< unsigned short > FromString< std::vector< unsigned short > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< unsigned int   > FromString< std::vector< unsigned int   > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< unsigned long  > FromString< std::vector< unsigned long  > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::vector< bool           > FromString< std::vector< bool           > > (const std::string& value );
    /// Parse larcv::PSet configuration file content
    template<> std::pair< int, int > FromString< std::pair< int, int > > (const std::string& value);
    /// Parse larcv::PSet configuration file content
    template <class T> std::string ToString(const T& value)
    { return std::to_string(value); }
    /// Parse larcv::PSet configuration file content
    template<> std::string ToString<std::string>(const std::string& value);
    /// Parse larcv::PSet configuration file content
    template <class T> std::string VecToString(const std::vector<T>& value)
    {
      std::string res="[";
      for(auto const& v : value)
	res += ToString(v) + ",";
      res = res.substr(0,res.size()-1);
      res += "]";
      return res;
    }
  }
}

#endif
