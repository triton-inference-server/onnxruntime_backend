if(DEFINED TRITON_ONNXRUNTIME_PACKAGE_URL)

  set(DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/_deps/downloads/onnxruntime.zip")
  set(EXTRACT_DIR "${CMAKE_BINARY_DIR}/_deps/downloads/onnxruntime")

  message(NOTICE "Downloading onnxruntime: ${TRITON_ONNXRUNTIME_PACKAGE_URL}")

  file(DOWNLOAD ${TRITON_ONNXRUNTIME_PACKAGE_URL} ${DOWNLOAD_PATH} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)

  # file(DOWNLOAD ... STATUS DOWNLOAD_STATUS) returns a list with 2 elements
  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)

  if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(NOTICE "Failed to download: ${TRITON_ONNXRUNTIME_PACKAGE_URL}")
  else()
    message(NOTICE "Download successful: ${DOWNLOAD_PATH}" )

    file(ARCHIVE_EXTRACT INPUT ${DOWNLOAD_PATH} DESTINATION ${EXTRACT_DIR} VERBOSE )

    file(READ "${EXTRACT_DIR}/VERSION_NUMBER" DOWNLOADED_ONNXRUNTIME_VERSION)
    if(${DOWNLOADED_ONNXRUNTIME_VERSION} VERSION_EQUAL ${TRITON_BUILD_ONNXRUNTIME_VERSION})
      message(NOTICE "Downloaded onnxruntime version: ${DOWNLOADED_ONNXRUNTIME_VERSION}")
      set(TRITON_ONNXRUNTIME_INCLUDE_PATHS ${EXTRACT_DIR}/include)
      set(TRITON_ONNXRUNTIME_LIB_PATHS ${EXTRACT_DIR}/lib)
    else()
      message(NOTICE "Downloaded onnxruntime version: ${DOWNLOADED_ONNXRUNTIME_VERSION} does not match the required version: ${TRITON_BUILD_ONNXRUNTIME_VERSION}")
    endif()

  endif(NOT DOWNLOAD_RESULT EQUAL 0)

endif(DEFINED TRITON_ONNXRUNTIME_PACKAGE_URL)
