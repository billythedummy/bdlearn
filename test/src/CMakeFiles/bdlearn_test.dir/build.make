# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake3

# The command to remove a file.
RM = /usr/bin/cmake3 -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /homes/iws/jula99/455/bdlearn/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /homes/iws/jula99/455/bdlearn/test/src

# Include any dependencies generated for this target.
include CMakeFiles/bdlearn_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bdlearn_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bdlearn_test.dir/flags.make

CMakeFiles/bdlearn_test.dir/main.o: CMakeFiles/bdlearn_test.dir/flags.make
CMakeFiles/bdlearn_test.dir/main.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/homes/iws/jula99/455/bdlearn/test/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bdlearn_test.dir/main.o"
	/opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bdlearn_test.dir/main.o -c /homes/iws/jula99/455/bdlearn/test/src/main.cpp

CMakeFiles/bdlearn_test.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bdlearn_test.dir/main.i"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /homes/iws/jula99/455/bdlearn/test/src/main.cpp > CMakeFiles/bdlearn_test.dir/main.i

CMakeFiles/bdlearn_test.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bdlearn_test.dir/main.s"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /homes/iws/jula99/455/bdlearn/test/src/main.cpp -o CMakeFiles/bdlearn_test.dir/main.s

CMakeFiles/bdlearn_test.dir/BMat_basic.o: CMakeFiles/bdlearn_test.dir/flags.make
CMakeFiles/bdlearn_test.dir/BMat_basic.o: BMat_basic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/homes/iws/jula99/455/bdlearn/test/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/bdlearn_test.dir/BMat_basic.o"
	/opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bdlearn_test.dir/BMat_basic.o -c /homes/iws/jula99/455/bdlearn/test/src/BMat_basic.cpp

CMakeFiles/bdlearn_test.dir/BMat_basic.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bdlearn_test.dir/BMat_basic.i"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /homes/iws/jula99/455/bdlearn/test/src/BMat_basic.cpp > CMakeFiles/bdlearn_test.dir/BMat_basic.i

CMakeFiles/bdlearn_test.dir/BMat_basic.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bdlearn_test.dir/BMat_basic.s"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /homes/iws/jula99/455/bdlearn/test/src/BMat_basic.cpp -o CMakeFiles/bdlearn_test.dir/BMat_basic.s

CMakeFiles/bdlearn_test.dir/Halide_test.o: CMakeFiles/bdlearn_test.dir/flags.make
CMakeFiles/bdlearn_test.dir/Halide_test.o: Halide_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/homes/iws/jula99/455/bdlearn/test/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/bdlearn_test.dir/Halide_test.o"
	/opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bdlearn_test.dir/Halide_test.o -c /homes/iws/jula99/455/bdlearn/test/src/Halide_test.cpp

CMakeFiles/bdlearn_test.dir/Halide_test.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bdlearn_test.dir/Halide_test.i"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /homes/iws/jula99/455/bdlearn/test/src/Halide_test.cpp > CMakeFiles/bdlearn_test.dir/Halide_test.i

CMakeFiles/bdlearn_test.dir/Halide_test.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bdlearn_test.dir/Halide_test.s"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /homes/iws/jula99/455/bdlearn/test/src/Halide_test.cpp -o CMakeFiles/bdlearn_test.dir/Halide_test.s

# Object files for target bdlearn_test
bdlearn_test_OBJECTS = \
"CMakeFiles/bdlearn_test.dir/main.o" \
"CMakeFiles/bdlearn_test.dir/BMat_basic.o" \
"CMakeFiles/bdlearn_test.dir/Halide_test.o"

# External object files for target bdlearn_test
bdlearn_test_EXTERNAL_OBJECTS =

bdlearn_test: CMakeFiles/bdlearn_test.dir/main.o
bdlearn_test: CMakeFiles/bdlearn_test.dir/BMat_basic.o
bdlearn_test: CMakeFiles/bdlearn_test.dir/Halide_test.o
bdlearn_test: CMakeFiles/bdlearn_test.dir/build.make
bdlearn_test: CMakeFiles/bdlearn_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/homes/iws/jula99/455/bdlearn/test/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable bdlearn_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bdlearn_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bdlearn_test.dir/build: bdlearn_test

.PHONY : CMakeFiles/bdlearn_test.dir/build

CMakeFiles/bdlearn_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bdlearn_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bdlearn_test.dir/clean

CMakeFiles/bdlearn_test.dir/depend:
	cd /homes/iws/jula99/455/bdlearn/test/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /homes/iws/jula99/455/bdlearn/test /homes/iws/jula99/455/bdlearn/test /homes/iws/jula99/455/bdlearn/test/src /homes/iws/jula99/455/bdlearn/test/src /homes/iws/jula99/455/bdlearn/test/src/CMakeFiles/bdlearn_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bdlearn_test.dir/depend

