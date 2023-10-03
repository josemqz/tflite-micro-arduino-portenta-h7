/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
Also based on ARM-software/developer repository, which provides 
methods to crop and scale the images from the Vision Shield.
https://github.com/ARM-software/developer/blob/master/projects/portenta_person_detection/person_detection/person_detection.ino
==============================================================================
*/

#include <TensorFlowLite.h>

#include "camera.h"
#include <SDRAM.h>

#include "detection_responder.h"
#include "image_provider.h"
#include "main_functions.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "SDMMCBlockDevice.h"
#include "FATFileSystem.h"

#include "ImageCropper.h"
#include "ImageScaler.h"

int image_count = 0; // affects tensor_arena alignment, but doesn't behave properly inside namespace

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  
  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.
  
  // An area of memory to use for input, output, and intermediate arrays.
  //constexpr int kTensorArenaSize = 2 * 1024 * 1024;
  constexpr int kTensorArenaSize = 136 * 1024;
  // Keep aligned to 16 bytes for CMSIS
  alignas(16) uint8_t tensor_arena[kTensorArenaSize]; // internal memory
  //uint8_t* tensor_arena = (uint8_t*) SDRAM_START_ADDRESS; // image memory assigned first, then tensor_arena

  // SD Card block device and filesystem
  SDMMCBlockDevice block_device;
  mbed::FATFileSystem fs("fs");

  constexpr bool write_serial_info = true;
  constexpr bool use_sd_card = true;
  constexpr int images_to_write = 5;
  
  constexpr int pd_large_image_width = 320;
  constexpr int pd_large_image_height = 240;
  constexpr int pd_cropped_dimension = 240;
  constexpr int pd_cropped_size = pd_cropped_dimension * pd_cropped_dimension;

  // Buffer for captured images (FrameBuffer is a class imported from camera.h)
  FrameBuffer largeImage(pd_large_image_width, pd_large_image_height, 2);
  uint8_t croppedImage[pd_cropped_size];
  uint8_t scaledImage[kMaxImageSize]; //kMaxImageSize from model_settings.h
  
  boolean sd_card_initialized = false;
  
  char filename[255];

  ImageCropper image_cropper;
  ImageScaler image_scaler;
    
}  // namespace

boolean init_sd_card()
{
  delay(2000);

  int err =  fs.mount(&block_device);
  if (err) {
    // Reformat if we can't mount the filesystem
    // this should only happen on the first boot
    Serial.println("formatting sd card");
    err = fs.reformat(&block_device);

    if(err) {
      Serial.print("failed to mount or reformat file system - error code = ");
      Serial.println(err);
      return false;
    }

    Serial.println("init_sd_card done formatting sd card");
  }

  return true;
}


void write_image_to_sd_card(const char* image_type, const uint8_t* image_data, int imageWidth, int imageHeight, int imageSize, int imageNum, const char *recognition)
{
    // form the file name
    sprintf(filename, "/fs/image_%s_%d_%dx%d_%d_%s.raw", image_type, imageNum, imageWidth, imageHeight, imageSize, recognition);

    if(write_serial_info) {
        Serial.print("writing ");
        Serial.print(imageSize);
        Serial.print(" bytes to SD card as ");
        Serial.println(filename);
    }

    FILE *file = fopen(filename, "wb");
    if(file != NULL) {
        size_t chunk_size = 512;
        size_t total_bytes_written = 0;
        while(total_bytes_written < (size_t) imageSize) {
             size_t write_size = min(chunk_size, imageSize - total_bytes_written);
             size_t bytes_written = fwrite(image_data + total_bytes_written, 1, write_size, file);
             total_bytes_written += bytes_written;
             if(bytes_written == 0)  {
                fclose(file);
                Serial.print("failed to write file ");
                Serial.println(filename);
                return;
             }
             if(bytes_written != write_size) {
                Serial.print("short write - attempted ");
                Serial.print(chunk_size);
                Serial.print(", wrote only ");
                Serial.println(bytes_written);
             }

             delay(20);
        }

        fclose(file);

        if(write_serial_info) {
            Serial.print("wrote ");
            Serial.print(total_bytes_written);
            Serial.println(" bytes to SD Card");
        }
    } else {
        Serial.print("fopen failed with error ");
        Serial.println(errno);
    }
}


// The name of this function is important for Arduino compatibility.
void setup() {

  // initialize SDRAM after allocated space for tensor_arena and some extra, just in case
  //SDRAM.begin(SDRAM_START_ADDRESS + kTensorArenaSize + 2 * 1024 * 1024);
  
  tflite::InitializeTarget();

  if(write_serial_info) {
    Serial.begin(115200);
    while (!Serial) {};
  }

  Serial.println("ARM Portenta H7 TensorFlow Person Detection Demo\n");

  if(use_sd_card && write_serial_info) {
    
      Serial.println("Initializing SD card...");
  }

  if(use_sd_card) {
      if (!init_sd_card() && write_serial_info) {
          Serial.println("SD card initialization failed, won't write images");
      } else {
          sd_card_initialized = true;

          if(write_serial_info) {
              Serial.println("SD card initialization done. Images will be written");
          }
      }
  }
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(
        "Model provided is schema version %d not equal "
        "to supported version ");
    Serial.println(model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(model, 
                                                    micro_op_resolver, 
                                                    tensor_arena,
                                                    kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  if ((input->dims->size != 4) || (input->dims->data[0] != 1) ||
      (input->dims->data[1] != kNumRows) ||
      (input->dims->data[2] != kNumCols) ||
      (input->dims->data[3] != kNumChannels) || (input->type != kTfLiteInt8)) {
    Serial.println("Bad input tensor parameters in model");
    return;
  }
}

// The name of this function is important for Arduino compatibility.
void loop() {

  if(write_serial_info) {
      Serial.println();
      Serial.println("==============================");
      Serial.println();
      Serial.print("Getting image ");
      Serial.print(image_count);
      Serial.print(" (");
      Serial.print(pd_large_image_width);
      Serial.print("x");
      Serial.print(pd_large_image_height);
      Serial.println(") from camera");
  }

  if (kTfLiteOk != GetImage(pd_large_image_width, pd_large_image_height, &largeImage)) {
    Serial.println("Image capture failed.");
  }

  // crop the image to the square aspect ratio that the model expects (will crop to center)
  if(write_serial_info) {
      Serial.print("Cropping image to ");
      Serial.print(pd_cropped_dimension);
      Serial.print("x");
      Serial.println(pd_cropped_dimension);
  }

  image_cropper.crop_image(largeImage.getBuffer(),
                          pd_large_image_width, 
                          pd_large_image_height, 
                          croppedImage, 
                          pd_cropped_dimension, 
                          pd_cropped_dimension);


  // scale the image to the size that the model expects (96x96)
  if(write_serial_info) {
      Serial.print("Scaling image to ");
      Serial.print(kNumCols);
      Serial.print("x");
      Serial.println(kNumRows);
  }
  int scale_result = image_scaler.scale_image_down(croppedImage, pd_cropped_dimension, pd_cropped_dimension, scaledImage, kNumRows, kNumCols);
  if(scale_result < 0) {
      Serial.println("Failed to scale image");
      return;
  }

  // copy the scaled image to the TF input
  if(write_serial_info) {
      Serial.println("Copying scaled image to TensorFlow model");
  }
  memcpy(input->data.uint8, scaledImage, kMaxImageSize);

  // Run the model on this input and make sure it succeeds.
  if(write_serial_info) {
      Serial.println("Invoking the TensorFlow interpreter");
  }
  
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    Serial.println("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float no_person_score_f =
      (no_person_score - output->params.zero_point) * output->params.scale;
  RespondToDetection(person_score_f, no_person_score_f);

  // write out the scaled image to the SD Card if present
  if(sd_card_initialized && image_count < images_to_write) {
     if(write_serial_info) {
          Serial.print("Writing scaled image ");
          Serial.print(image_count);
          Serial.println(" to sd card");
      }

      write_image_to_sd_card("scaled", scaledImage, kNumRows, kNumCols, kMaxImageSize, image_count, person_score > no_person_score ? "PERSON" : "NOPERSON");
  }
}
