/**
 * @file search_factory.hpp
 * @brief Picks the search backend once, at WayComputer construction.
 */

#pragma once

#include <memory>

#include "modules/isearch.hpp"
#include "utils/Params.hpp"

/**
 * @brief Builds the backend named by params.search_backend ("cpu", "cpu-fast",
 * "cuda" or "cuda-one-shot"), falling back to cpu-fast when it cannot be --
 * no device, no USE_CUDA, a configuration the device cannot represent, or an
 * unrecognised name. Every fallback logs why.
 *
 * The choice is always logged, because "which backend am I running" is the
 * first question whenever the numbers look wrong.
 */
std::unique_ptr<ISearch> createSearch(const Params::WayComputer &params);
