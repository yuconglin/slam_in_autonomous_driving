add_executable(motion motion.cc)
target_link_libraries(motion
        ${PROJECT_NAME}.common
        ${PROJECT_NAME}.tools
        )

add_executable(homework_ch2_3 homework_ch2_3.cc)
target_link_libraries(homework_ch2_3
        ${PROJECT_NAME}.common
        ${PROJECT_NAME}.tools
        )
