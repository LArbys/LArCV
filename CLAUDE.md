# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

LArCV uses CMake as its primary build system. The repository includes both traditional configure/make scripts and modern CMake.

### Basic Build Process
```bash
# Configure environment (sets up paths and detects dependencies)
source configure.sh

# Create build directory and compile
cd $LARCV_BUILDDIR  # Usually build/
make

# Or using CMake directly:
mkdir build && cd build
cmake .. -DUSE_PYTHON3=ON -DUSE_OPENCV=ON
make install
```

### Build Options
Key CMake options that can be set:
- `USE_PYTHON3=ON/OFF` - Build Python 3 bindings (default OFF)
- `USE_PYTHON2=ON/OFF` - Build Python 2 bindings (default OFF) 
- `USE_OPENCV=ON/OFF` - Enable OpenCV support (auto-detected)
- `USE_TORCH=ON/OFF` - Enable PyTorch C++ API support
- `USE_GEO2D=ON/OFF` - Enable GEO2D geometry package

### Environment Detection
The build system auto-detects capabilities through environment variables:
- `OPENCV_INCDIR/OPENCV_LIBDIR` - OpenCV paths
- `LIBTORCH_INCDIR/LIBTORCH_LIBDIR` - PyTorch paths  
- `GEO2D_BASEDIR` - GEO2D geometry package
- `NLOHMANN_JSON_DIR` - JSON library (falls back to bundled version)

### Key Build Paths
After running `configure.sh`:
- `$LARCV_BUILDDIR` - Build directory (usually `build/`)
- `$LARCV_LIBDIR` - Libraries (`build/installed/lib/`)
- `$LARCV_INCDIR` - Headers (`build/installed/include/`)
- `$LARCV_BINDIR` - Executables (`build/installed/bin/`)

## Architecture Overview

LArCV is a framework for processing Liquid Argon Time Projection Chamber (LArTPC) detector data, designed as a bridge between LArSoft and deep learning frameworks.

### Core Framework Structure

**larcv/core/** - Foundation components:
- **Base/**: Infrastructure (logging, configuration, base classes)
- **DataFormat/**: Core data structures and I/O management
- **Processor/**: Plugin-based processing framework
- **PyUtil/**: Python/NumPy integration
- **CVUtil/**: OpenCV integration
- **TorchUtil/**: PyTorch integration
- **json/**: JSON utilities

**larcv/app/** - Application-specific processors:
- **ImageMod/**: Image processing algorithms
- **Filter/**: Event filtering
- **MeatSlicer/**: LArSoft integration and data conversion
- **LArOpenCVHandle/**: Computer vision algorithms

### Key Data Types

**Image Data:**
- `Image2D`: 2D detector images with pixel data in column-major order
- `ImageMeta`: Coordinate system mapping between pixel (row,col) and physical (wire,tick) coordinates
- `EventImage2D`: Event container for multiple Image2D objects (different wire planes)

**3D Data:**
- `Voxel3D`: 3D voxel elements (ID + value)
- `SparseTensor3D`: Sparse 3D tensor container
- `Voxel3DMeta`: 3D coordinate system metadata

**Physics Data:**
- `Particle`: Physics particle information (energy, momentum, PDG code)
- `ROI`: Region of Interest with bounding boxes and classifications
- `PGraph`: Graph structures for particle relationships

### Processing Framework Pattern

LArCV uses a **plugin-based processor architecture**:

1. **ProcessBase**: Abstract base class for all algorithms
   - `configure()`: Initialize from parameters
   - `process(IOManager&)`: Per-event processing
   - `initialize()/finalize()`: Setup/cleanup

2. **ProcessDriver**: Orchestrates multiple processors
   - Manages event loops and data flow
   - Configuration-driven processor selection

3. **IOManager**: Handles all file I/O
   - Automatic ROOT tree management
   - Type-safe data product access

### Data Flow
```
Input Files → IOManager → ProcessDriver → [ProcessBase plugins] → Output Files
```

### Important Concepts

- **Event-based processing**: All data inherits from `EventBase` with run/subrun/event IDs
- **Coordinate systems**: Flexible mapping between detector and image coordinates
- **Multi-framework support**: Integrates with ROOT, OpenCV, PyTorch, and Python
- **Memory efficiency**: Sparse representations and move semantics for large datasets
- **Configuration-driven**: Processors instantiated and configured via parameter files

### Python Integration

Python bindings provide direct access to C++ data structures:
- Image2D ↔ NumPy arrays (with automatic memory management)
- Event-level data access through PyUtil classes
- Integration with data loading for ML frameworks