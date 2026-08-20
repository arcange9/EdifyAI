#!/usr/bin/env bash
# Edify AI — Release Helper
# Usage:
#   ./scripts/release.sh 1.2.0        # Creates tag v1.2.0 and pushes (triggers release workflow)
#   ./scripts/release.sh 1.2.0 --pre  # Creates a pre-release tag

set -euo pipefail

VERSION="${1:-}"
PRERELEASE="${2:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: ./scripts/release.sh <version> [--pre]"
  echo "Example: ./scripts/release.sh 1.2.0"
  exit 1
fi

TAG="v${VERSION}"

echo "Updating package.json version to $VERSION..."
sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"$VERSION\"/" package.json
rm -f package.json.bak

git add package.json
echo "Creating tag $TAG..."
git commit -m "release: $TAG"
git tag -a "$TAG" -m "Edify AI $VERSION"

echo "Pushing to origin..."
git push origin main
git push origin "$TAG"

echo "Done! The release workflow will build and publish the .exe automatically."
echo "Monitor: https://github.com/arcange9/EdifyAI/actions"
echo "Release: https://github.com/arcange9/EdifyAI/releases/tag/$TAG"
