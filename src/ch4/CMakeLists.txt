add_library(${PROJECT_NAME}.ch4
        gins_pre_integ.cc
        imu_preintegration.cc
        g2o_types.cc
        )

ADD_EXECUTABLE(test_preintegration test_preintegration.cc)
ADD_TEST(test_preintegration test_preintegration)

target_link_libraries(test_preintegration
        gtest pthread glog gflags ${PROJECT_NAME}.ch4 ${PROJECT_NAME}.ch3 ${PROJECT_NAME}.common
        )

add_library(ceres_lib
   pose_local_parameterization.cc
)
target_link_libraries(ceres_lib ${CERES_LIBRARIES})

ADD_EXECUTABLE(test_preintegration_ceres test_preintegration_ceres.cc)
ADD_TEST(test_preintegration_ceres test_preintegration_ceres)
target_link_libraries(test_preintegration_ceres
        gtest pthread glog gflags ${PROJECT_NAME}.ch4 ${PROJECT_NAME}.ch3 ${PROJECT_NAME}.common
        ceres_lib ${CERES_LIBRARIES})

ADD_EXECUTABLE(test_preintegration_new test_preintegration_new.cc)
ADD_TEST(test_preintegration_new test_preintegration_new)
target_link_libraries(test_preintegration_new
        gtest pthread glog gflags ${PROJECT_NAME}.ch4 ${PROJECT_NAME}.ch3 ${PROJECT_NAME}.common
        )

add_executable(run_gins_pre_integ run_gins_pre_integ.cc)
target_link_libraries(run_gins_pre_integ
        ${PROJECT_NAME}.ch3
        ${PROJECT_NAME}.ch4
        ${g2o_libs}
        )