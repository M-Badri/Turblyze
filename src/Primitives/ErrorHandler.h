/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file ErrorHandler.h
 * @brief Fatal error and warning for program diagnostics
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <iostream>
#include <source_location>

// Project headers
#include "StringTypes.h"
#include "Comm.h"

// *********************************** Alias **********************************

using Location = std::source_location;

// ************************* Error Handling Functions *************************

/// Print a fatal error message and abort the program
[[noreturn]] inline void FatalError
(
    const Message& errorMessage,
    const Location errorLocation = Location::current()
) noexcept
{
    std::cerr
        << '\n' << '\n' << "FATAL ERROR"
        << '\n' << "    " << errorLocation.file_name() << ':'
        << errorLocation.line()
        << '\n' << "    " << errorMessage << '\n' << '\n' << std::endl;

    Comm::abort(1);
}


/// Print a warning message and continue the program
inline void Warning
(
    const Message& warningMessage,
    const Location warningLocation = Location::current()
) noexcept
{
    std::cerr
        << '\n' << "[WARNING] (" << warningLocation.file_name() << ':'
        << warningLocation.line() << ") " << warningMessage << std::endl;
}
