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

rm -rf _site
mkdir -p _site

##### Get potential new docs from gh-pages #####
if git ls-remote --heads origin gh-pages | grep -q gh-pages; then
    echo "gh-pages branch exists, setting up worktree for sync..."
    git fetch origin gh-pages:gh-pages
    git worktree add gh-pages gh-pages
    rsync -av --ignore-existing gh-pages/ _site/
    git worktree remove gh-pages --force
else
    echo "gh-pages branch does not exist, skipping sync."
fi

##### Get archived docs #####
if git ls-remote --heads origin doc_archive | grep -q doc_archive; then
    echo "doc_archive branch exists, syncing archived versions..."
    git fetch origin doc_archive:doc_archive
    git worktree add archive_sync doc_archive
    rsync -av --ignore-existing archive_sync/ _site/
    git worktree remove archive_sync --force
else
    echo "doc_archive branch does not exist, skipping archive sync."
fi

##### Prepare new doc #####
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

##### ARCHIVE NEW VERSION #####
ARCHIVE_WORKTREE="archive"
STORAGE_BRANCH="doc_archive"
echo "Archiving version $SHORT_DOC_VERSION to branch $STORAGE_BRANCH..."
# Fetch or create archive branch
if git ls-remote --heads origin "$STORAGE_BRANCH" | grep -q "$STORAGE_BRANCH"; then
    git fetch origin $STORAGE_BRANCH:$STORAGE_BRANCH
else
    git branch --orphan $STORAGE_BRANCH
fi

git worktree add $ARCHIVE_WORKTREE $STORAGE_BRANCH

mkdir -p $ARCHIVE_WORKTREE/$SHORT_DOC_VERSION
cp -R _site/$SHORT_DOC_VERSION/* $ARCHIVE_WORKTREE/$SHORT_DOC_VERSION/

# Commit & push
pushd $ARCHIVE_WORKTREE >/dev/null
  git config user.name "github-actions[bot]"
  git config user.email "github-actions[bot]@users.noreply.github.com"
  git add $SHORT_DOC_VERSION
  if ! git diff --staged --quiet; then
    git commit -m "Archive docs version $SHORT_DOC_VERSION"
    git push origin $STORAGE_BRANCH
  else
    echo "No changes to archive for $SHORT_DOC_VERSION"
  fi
popd >/dev/null

# Cleanup archive worktree
git worktree remove $ARCHIVE_WORKTREE --force
