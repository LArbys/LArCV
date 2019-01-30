# Features in new version

Here we list and describe the big changes to the core of how this library works or is built.

* we aim to build with cmake rather than our custom set of Makefiles.
  this is to help incorporate the library into other projects,
  as cmake is the usual build manager for most projects.

* we do leave the old makefile system lying around, however. but one should consider this deprecated.

* headers are now installed in a `larcv` folder. This means that `#include` statements need to be updated.
  For example, `#include "Base/DataFormat/Image2D.h"` now should be `#include "larcv/Base/DataFormat/Image2D.h"`.
  This follows more standard practices. It also helps the reader know that the header comes from larcv.
  This is annoying to change. So we provide the a script, `misc/patch.sh` to help migrate code.
  It uses `sed` to replace known modules. 