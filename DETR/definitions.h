#pragma once

#ifndef LIBS_DEFINITIONS_H
#define LIBS_DEFINITIONS_H

#ifndef _IOSTREAM_
#include <iostream>
#endif

#define MESSAGE_LOG(message) (std::cout << (message) << " " << __FUNCTION__ << " " << __FILE__ << " " << __LINE__ << std::endl);
#define MESSAGE_LOG_ObJ(message, object) (std::cout << (message) << " " << (object) << " " << __FUNCTION__ << " " << __FILE__ << " " << __LINE__ << std::endl);

#endif