#include "memvanta/mmap_file.hpp"
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace memvanta {
MMapFile::MMapFile(const std::string& path){ open(path); }
MMapFile::~MMapFile(){ close(); }
MMapFile::MMapFile(MMapFile&& o) noexcept : fd_(o.fd_), data_(o.data_), size_(o.size_){ o.fd_=-1; o.data_=nullptr; o.size_=0; }
MMapFile& MMapFile::operator=(MMapFile&& o) noexcept { if(this!=&o){ close(); fd_=o.fd_; data_=o.data_; size_=o.size_; o.fd_=-1;o.data_=nullptr;o.size_=0;} return *this; }
void MMapFile::open(const std::string& path){ close(); fd_=::open(path.c_str(), O_RDONLY); if(fd_<0) throw std::runtime_error("open failed: "+path); struct stat st{}; if(fstat(fd_,&st)!=0){ close(); throw std::runtime_error("fstat failed"); } size_=st.st_size; if(size_==0){ close(); throw std::runtime_error("empty file"); } void* p=mmap(nullptr,size_,PROT_READ,MAP_PRIVATE,fd_,0); if(p==MAP_FAILED){ close(); throw std::runtime_error("mmap failed"); } data_=static_cast<std::byte*>(p); madvise(data_,size_,MADV_RANDOM); }
void MMapFile::close(){ if(data_){ munmap(data_,size_); data_=nullptr;} if(fd_>=0){::close(fd_);fd_=-1;} size_=0; }
void MMapFile::advise_sequential(std::uint64_t off,std::uint64_t len) const { if(data_ && off<size_) madvise(data_+off, std::min<std::uint64_t>(len,size_-off), MADV_SEQUENTIAL); }
void MMapFile::advise_willneed(std::uint64_t off,std::uint64_t len) const { if(data_ && off<size_) madvise(data_+off, std::min<std::uint64_t>(len,size_-off), MADV_WILLNEED); }
void MMapFile::advise_dontneed(std::uint64_t off,std::uint64_t len) const { if(data_ && off<size_) madvise(data_+off, std::min<std::uint64_t>(len,size_-off), MADV_DONTNEED); }
}
