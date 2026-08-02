/**
 * @file search_factory.hpp
 * @brief Picks the search backend once, at WayComputer construction.
 */

#pragma once

#include <memory>

#include "modules/isearch.hpp"
#include "utils/Params.hpp"

/**
 * @brief Returns the best available backend: the CUDA one when the package was
 * built with USE_CUDA and a device is actually present, the CPU one otherwise.
 * The choice is logged, because "which backend am I running" is the first
 * question whenever the numbers look wrong.
 */
std::unique_ptr<ISearch> createSearch(const Params::WayComputer &params);
