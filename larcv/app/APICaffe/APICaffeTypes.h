#ifndef APICAFFETYPES_H
#define APICAFFETYPES_H

namespace larcv {

	enum ThreadFillerState_t {
		kThreadStateIdle,
		kThreadStateStarting,
		kThreadStateRunning,
		kThreadStateUnknown
	};

}

#endif