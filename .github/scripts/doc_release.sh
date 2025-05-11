#!/bin/bash

#===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Ensure the build directory exists
BUILD_DIR="doc/_build/scikit-learn-intelex"
if [ ! -d "$BUILD_DIR" ]; then
    echo "::error: Documentation build directory not found!"
    exit 1
fi

mkdir -p _site
        
# sync only new version folders from gh-pages into _site
rsync -av --ignore-existing gh-pages/ _site/

# Copy the new built version to _site
mkdir -p _site/$SHORT_DOC_VERSION
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* _site/$SHORT_DOC_VERSION/

# Update latest
rm -rf _site/latest
mkdir -p _site/latest
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* _site/latest/

# Copy index.html
cp doc/_build/scikit-learn-intelex/index.html _site/

# Generate versions.json
mkdir -p _site/doc
echo "[" > _site/doc/versions.json
# Add latest entry first
echo '  {"name": "latest", "version": "'$SHORT_DOC_VERSION'", "url": "/scikit-learn-intelex/latest/"},' >> _site/doc/versions.json
# Add all year.month folders
for version in $(ls -d _site/[0-9][0-9][0-9][0-9].[0-9]* 2>/dev/null || true); do
    version=$(basename "$version")
    echo '  {"name": "'$version'", "version": "'$version'", "url": "/scikit-learn-intelex/'$version'/"},'
done | sort -rV >> _site/doc/versions.json
# Remove trailing comma and close array
sed -i '$ s/,$//' _site/doc/versions.json
echo "]" >> _site/doc/versions.json

# Display the content for verification
ls -la _site/
cat _site/doc/versions.json