# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os


# Prepare dataset

# OBS: when 800 all images are used it results in 482070 sliced images
# OBS: that gives 482070/800 ~= 602 sliced images per image
# OBS: using 20 images then gives around 12k training images
# OBS: using 50 images then gives around 30k training images
# OBS: there were 2837 images in original training data

os.system("python3 ./prepare_dataset.py --images_dir /work3/s204163/data/ImageNet/original --output_dir /work3/s204163/data/ImageNet/SRGAN/train --image_size 128 --step 64 --num_workers 16 --num_images 50")