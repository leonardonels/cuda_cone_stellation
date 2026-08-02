cmake_minimum_required(VERSION 3.5)
project(cuda_cone_stellation)

if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Search scalar, chosen PER BACKEND. This used to be a single package-wide
# switch on the grounds that both backends had to agree; that forced the host
# and the device to make the same choice when they want opposite ones (Orin's
# FP64 rate is ~1/32 of its FP32 rate, so a device backend wants float, while
# the reference host backend wants whatever keeps it a faithful reference).
# What the backends must actually share is the POLICY -- the filters in
# structures/search_policy.hpp -- not the width they evaluate it in.
# The published Way stays double either way: this only affects the search's own
# arithmetic.
option(CCS_CPU_SCALAR_FLOAT32 "Run the CPU tree search in float32" ON)

# The package-wide flag is gone. Inherit its value rather than silently
# switching the scalar out from under an existing build directory.
if(DEFINED CCS_SCALAR_FLOAT32)
	message(WARNING
		"CCS_SCALAR_FLOAT32 has been replaced by per-backend flags. Inheriting its "
		"value (${CCS_SCALAR_FLOAT32}) for CCS_CPU_SCALAR_FLOAT32; use "
		"-DCCS_CPU_SCALAR_FLOAT32 from now on.")
	set(CCS_CPU_SCALAR_FLOAT32 ${CCS_SCALAR_FLOAT32} CACHE BOOL "" FORCE)
	unset(CCS_SCALAR_FLOAT32 CACHE)
endif()

if(CCS_CPU_SCALAR_FLOAT32)
	add_compile_definitions(CCS_CPU_SCALAR_FLOAT32)
	message(STATUS "cuda_cone_stellation: cpu search scalar = float32")
else()
	message(STATUS "cuda_cone_stellation: cpu search scalar = double")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(mmr_base REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Include directories
include_directories(
  include/cuda_cone_stellation
  ${EIGEN3_INCLUDE_DIRS}
)

#---------------------------------------------------------
#                            ACCELERATION
#---------------------------------------------------------

add_library(acceleration src/acceleration.cpp)

ament_target_dependencies(acceleration
	rclcpp
	geometry_msgs
	mmr_base
	nav_msgs
	visualization_msgs
)

#---------------------------------------------------------
#                             AUTOCROSS
#---------------------------------------------------------

add_library(autocross
  src/autocross.cpp
  src/modules/DelaunayTri.cpp
  src/modules/Visualization.cpp
  src/modules/WayComputer.cpp
  src/modules/cpu_fast_search.cpp
  src/modules/cpu_search.cpp
  src/modules/isearch.cpp
  src/modules/search_factory.cpp
  src/structures/Circle.cpp
  src/structures/Edge.cpp
  src/structures/Node.cpp
  src/structures/Point.cpp
  src/structures/Triangle.cpp
  src/structures/Vector.cpp
  src/structures/Way.cpp
  src/utils/Failsafe.cpp
  src/utils/KDTree.cpp
  src/utils/Params.cpp
  src/utils/Time.cpp
)

ament_target_dependencies(autocross
	rclcpp
	geometry_msgs
	nav_msgs
	visualization_msgs
	mmr_base
	Eigen3
	tf2_eigen
	tf2_geometry_msgs
)

#---------------------------------------------------------
#                             SKIDPAD
#---------------------------------------------------------

add_library(skidpad src/skidpad.cpp)

ament_target_dependencies(skidpad
	rclcpp
	geometry_msgs
	mmr_base
	nav_msgs
	visualization_msgs
)

#---------------------------------------------------------
#                             NODE
#---------------------------------------------------------

add_executable(${PROJECT_NAME}_node
  		       src/${PROJECT_NAME}_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
					  acceleration
					  autocross
					  skidpad)

ament_target_dependencies(${PROJECT_NAME}_node
						  rclcpp
						  geometry_msgs
						  nav_msgs
						  mmr_base
						  visualization_msgs
						  rmw
						  )

#---------------------------------------------------------
#                       OFFLINE BENCHMARK
#---------------------------------------------------------
# Deterministic bag replay, used to compare search backends. OFF by default so
# the flight build neither compiles it nor takes a dependency on rosbag2.
#   colcon build --packages-select cuda_cone_stellation --cmake-args -DBUILD_BENCH=ON
option(BUILD_BENCH "Build the offline bag-replay benchmark" OFF)
if(BUILD_BENCH)
	find_package(rosbag2_cpp REQUIRED)
	find_package(rosbag2_storage REQUIRED)

	add_executable(search_bench bench/search_bench.cpp)
	target_link_libraries(search_bench autocross)
	ament_target_dependencies(search_bench
		rclcpp
		geometry_msgs
		nav_msgs
		mmr_base
		visualization_msgs
		rosbag2_cpp
		rosbag2_storage
	)
	install(TARGETS search_bench DESTINATION lib/${PROJECT_NAME})
endif()

#_________________________________________________________
#                             DIRECTORYS
#_________________________________________________________


install(TARGETS
	    acceleration
	    autocross
	    skidpad
	    ${PROJECT_NAME}_node
	    DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
        config
        DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY
	    launch
	    DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
	find_package(ament_lint_auto REQUIRED)
	ament_lint_auto_find_test_dependencies()
endif()

ament_package()