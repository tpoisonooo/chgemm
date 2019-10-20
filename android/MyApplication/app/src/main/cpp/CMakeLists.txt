# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

enable_language(ASM)

set(CPP_SRC_DIR "../../../../../../")

include_directories(${CPP_SRC_DIR})

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             native-lib.cpp
             ${CPP_SRC_DIR}reorder_a.S
             ${CPP_SRC_DIR}compare_matrices.c
             ${CPP_SRC_DIR}copy_matrix.c
             ${CPP_SRC_DIR}dclock.c
             ${CPP_SRC_DIR}int8kernel_m1.S
             ${CPP_SRC_DIR}int8kernel_m2.S
             ${CPP_SRC_DIR}int8kernel_m4.S
             ${CPP_SRC_DIR}print_matrix.c
             ${CPP_SRC_DIR}MMult_4x8_22.c
             ${CPP_SRC_DIR}print_matrix.c
             ${CPP_SRC_DIR}random_matrix.c
             ${CPP_SRC_DIR}REF_MMult.c
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )