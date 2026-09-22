/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Logger.cpp
 * @brief Implementations of solver-output formatting helpers
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "Logger.h"

// Standard library headers
#include <format>
#include <iostream>
#include <string>

// ***************************** namespace Logger *****************************

void Logger::sectionHeader(const Message& title)
{
    std::cout
        << "========================================"
        << "========================================"
        << '\n' << " " << title << '\n'
        << "----------------------------------------"
        << "----------------------------------------"
        << '\n';
}


void Logger::iterationHeader(Count n)
{
    sectionHeader(std::format("Iteration {}", n));
}


void Logger::iterationFooter()
{
    std::cout
        << "========================================"
        << "========================================"
        << '\n';
}


void Logger::linearSolverConfigHeader()
{
    std::cout << std::format
    (
        "    {:<10}{:<12}{:<16}{:<16}{:<10}\n"
        "    {:<10}{:<12}{:<16}{:<16}{:<10}\n",
        "Equation", "Solver", "Preconditioner", "Tolerance", "Max Iters",
        "--------", "------", "--------------", "---------", "---------"
    );
}


void Logger::linearSolverConfigRow
(
    const Name& equation,
    const Name& solver,
    const Name& preconditioner,
    Scalar tolerance,
    Count maxIters
)
{
    std::cout << std::format
    (
        "    {:<10}{:<12}{:<16}{:<16g}{:<10}\n",
        equation, solver, preconditioner, tolerance, maxIters
    );
}


void Logger::keyValue(const Message& label, Scalar value)
{
    std::cout << std::format
    (
        "    {:<24}  {:.6g}\n",
        label, value
    );
}


void Logger::keyValue(const Message& label, Scalar value, const Message& unit)
{
    std::cout << std::format
    (
        "    {:<24}  {:.6g} {}\n",
        label, value, unit
    );
}


void Logger::keyValue(const Message& label, Count value)
{
    std::cout << std::format
    (
        "    {:<24}  {}\n",
        label, value
    );
}


void Logger::keyValue(const Message& label, const Message& value)
{
    std::cout << std::format
    (
        "    {:<24}  {}\n",
        label, value
    );
}


void Logger::residualTableHeader()
{
    std::cout << std::format
    (
        "  {:<11} {:<11} {:>5}    {}\n"
        "  {:<11} {:<11} {:>5}    {}\n",
        "Equation", "Solver", "Iters", "Linear Solver Residual",
        "--------", "--------", "-----", "----------------------"
    );
}


void Logger::residualRow
(
    const Name& equation,
    const Name& solver,
    int iterations,
    Scalar linearSolverResidual
)
{
    std::cout << std::format
    (
        "  {:<11} {:<11} {:>5}    {:.6e}\n",
        equation, solver, iterations, linearSolverResidual
    );
}


void Logger::subsection(const Message& title)
{
    std::cout << '\n' << "  " << title << '\n';
}


void Logger::breakdownHeader(const Message& cornerLabel)
{
    std::cout << std::format
    (
        "\n  {:<14}{:>16}{:>16}{:>16}\n"
        "  {:<14}{:>16}{:>16}{:>16}\n",
        cornerLabel, "Pressure", "Friction", "Total",
        "----------", "--------", "--------", "-----"
    );
}


void Logger::breakdownRow
(
    const Message& label,
    Scalar pressure,
    Scalar friction,
    Scalar total
)
{
    std::cout << std::format
    (
        "  {:<14}{:>16.6e}{:>16.6e}{:>16.6e}\n",
        label, pressure, friction, total
    );
}


void Logger::scalarStat
(
    const Name& name,
    Scalar minVal,
    Scalar maxVal,
    Scalar meanVal
)
{
    std::cout << std::format
    (
        "    {:<7}min={:.2e}  max={:.2e}  mean={:.2e}\n",
        name, minVal, maxVal, meanVal
    );
}


void Logger::scaledResidual(const Name& name, Scalar value)
{
    std::cout << std::format
    (
        "    {:<10}{:.6e}\n",
        name, value
    );
}


void Logger::residualSummary
(
    Scalar mass,
    Scalar velocity,
    Scalar pressure,
    const std::vector<Residuals>& residuals
)
{
    std::string summary = std::format
    (
        " - Mass: {:.6e}, Velocity: {:.6e}, Pressure: {:.6e}",
        mass, velocity, pressure
    );

    for (const Residuals& residual : residuals)
    {
        summary += std::format(", {}: {:.6e}", residual.first, residual.second);
    }

    std::cout << summary << '\n';
}
