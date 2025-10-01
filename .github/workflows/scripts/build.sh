#!/usr/bin/env bash

# Get dependencies
echo "################"
echo "Get dependencies"
npm install

# Update git config
git config --global --add safe.directory /__w/learn-software-engineering/website

# Show tools versions
echo "####################"
echo "Running hugo version"
hugo version

echo "######################"
echo "Running sass --version"
sass --version

echo "################"
echo "Running hugo env"
hugo env

# Environment variables
# For maximum backward compatibility with Hugo modules
echo "###########################"
echo "Check environment variables"
if [[ -z "${ENV}" ]]; then
    echo "ENV should be set (development|production)"
    exit 1
else
    echo "Building website for $ENV"
fi
export HUGO_ENVIRONMENT=$ENV
export HUGO_ENV=$ENV

# Build site
echo "##########"
echo "Build site"
hugo --gc --minify --noBuildLock --enableGitInfo --cleanDestinationDir \
  --printPathWarnings --printI18nWarnings \
  --printMemoryUsage --printUnusedTemplates \
  --templateMetrics --templateMetricsHints \
  --environment $ENV

# Avoid GitHub Jekyll processing
echo "##############################"
echo "Avoid GitHub Jekyll processing"
touch public/.nojekyll

# Add CNAME config if it is production
echo "####################################"
echo "Add CNAME config if it is production"
if [ $ENV == "production" ]; then
  echo 'learn-software.com' > public/CNAME
fi
