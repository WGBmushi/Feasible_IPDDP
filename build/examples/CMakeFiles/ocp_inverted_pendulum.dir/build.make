# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/ocp_inverted_pendulum.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/ocp_inverted_pendulum.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/ocp_inverted_pendulum.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/ocp_inverted_pendulum.dir/flags.make

examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o: examples/CMakeFiles/ocp_inverted_pendulum.dir/flags.make
examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o: ../examples/ocp_inverted_pendulum.cpp
examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o: examples/CMakeFiles/ocp_inverted_pendulum.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o"
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o -MF CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o.d -o CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o -c /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/examples/ocp_inverted_pendulum.cpp

examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.i"
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/examples/ocp_inverted_pendulum.cpp > CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.i

examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.s"
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/examples/ocp_inverted_pendulum.cpp -o CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.s

# Object files for target ocp_inverted_pendulum
ocp_inverted_pendulum_OBJECTS = \
"CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o"

# External object files for target ocp_inverted_pendulum
ocp_inverted_pendulum_EXTERNAL_OBJECTS =

examples/ocp_inverted_pendulum: examples/CMakeFiles/ocp_inverted_pendulum.dir/ocp_inverted_pendulum.cpp.o
examples/ocp_inverted_pendulum: examples/CMakeFiles/ocp_inverted_pendulum.dir/build.make
examples/ocp_inverted_pendulum: libipddp.so
examples/ocp_inverted_pendulum: /usr/local/lib/libsymengine.a
examples/ocp_inverted_pendulum: /usr/lib/aarch64-linux-gnu/libgmp.so
examples/ocp_inverted_pendulum: /usr/lib/aarch64-linux-gnu/libpython3.10.so
examples/ocp_inverted_pendulum: examples/CMakeFiles/ocp_inverted_pendulum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ocp_inverted_pendulum"
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ocp_inverted_pendulum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/ocp_inverted_pendulum.dir/build: examples/ocp_inverted_pendulum
.PHONY : examples/CMakeFiles/ocp_inverted_pendulum.dir/build

examples/CMakeFiles/ocp_inverted_pendulum.dir/clean:
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/ocp_inverted_pendulum.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/ocp_inverted_pendulum.dir/clean

examples/CMakeFiles/ocp_inverted_pendulum.dir/depend:
	cd /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/examples /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples /home/parallels/Academic/Arsenal/DDP/IPDDP_SymEngine_Series/Feasible_IPDDP/build/examples/CMakeFiles/ocp_inverted_pendulum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/ocp_inverted_pendulum.dir/depend

