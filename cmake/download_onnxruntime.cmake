if(WIN32)

  message(NOTICE "Downloading onnxruntime: ${TRITON_ONNXRUNTIME_PACKAGE_URL}")

  # Define the download and extraction paths)
  set(DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/_deps/downloads/onnxruntime.zip")
  set(EXTRACT_DIR "${CMAKE_BINARY_DIR}/_deps/downloads/onnxruntime")

  # Download the file
  file(DOWNLOAD ${TRITON_ONNXRUNTIME_PACKAGE_URL} ${DOWNLOAD_PATH} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)

  # Check the download status
  list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
  if(NOT DOWNLOAD_RESULT EQUAL 0)
      message(SEND_ERROR "Failed to download: ${TRITON_ONNXRUNTIME_PACKAGE_URL}")
  else()
      message(NOTICE "Download successful: ${DOWNLOAD_PATH}" )

      # Extract the downloaded file
      file(ARCHIVE_EXTRACT INPUT ${DOWNLOAD_PATH} DESTINATION ${EXTRACT_DIR} VERBOSE )

      set(TRITON_ONNXRUNTIME_INCLUDE_PATHS ${EXTRACT_DIR}/include)
      set(TRITON_ONNXRUNTIME_LIB_PATHS ${EXTRACT_DIR}/lib)

  endif(NOT DOWNLOAD_RESULT EQUAL 0)

endif(WIN32)