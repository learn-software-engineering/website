#!/usr/bin/env bash

# This script submits Hugo site URLs to IndexNow and Bing APIs

# Get environment variables (set by GitHub Actions)
SITE_URL="${SITE_URL:-https://learn-software.com}"
API_KEY="${API_KEY:-53f1811377874f608f161d768a9c0b78}"
KEY_LOCATION="$SITE_URL/$API_KEY.txt"

# IndexNow endpoints
INDEXNOW_API="https://api.indexnow.org/indexnow"
BING_API="https://www.bing.com/indexnow"

# Function to submit URLs to IndexNow
submit_to_indexnow() {
    local urls_json="$1"

    echo "=== IndexNow Submission ==="
    echo "Site: $SITE_URL"
    echo "Timestamp: $(date)"
    echo

    # Submit to IndexNow API
    echo "Submitting to IndexNow API..."
    response1=$(curl -s -w "HTTP_STATUS:%{http_code}" -X POST \
        -H "Content-Type: application/json; charset=utf-8" \
        -d "$urls_json" \
        "$INDEXNOW_API")

    status1=$(echo "$response1" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    echo "IndexNow API response: HTTP $status1"

    # Submit to Bing API
    echo "Submitting to Bing API..."
    response2=$(curl -s -w "HTTP_STATUS:%{http_code}" -X POST \
        -H "Content-Type: application/json; charset=utf-8" \
        -d "$urls_json" \
        "$BING_API")

    status2=$(echo "$response2" | grep -o "HTTP_STATUS:[0-9]*" | cut -d: -f2)
    echo "Bing API response: HTTP $status2"

    # Check results
    if [[ "$status1" == "200" ]] || [[ "$status2" == "200" ]]; then
        echo "IndexNow submission successful"
        return 0
    else
        echo "IndexNow submission failed"
        echo "IndexNow status: $status1"
        echo "Bing status: $status2"
        return 1
    fi
}

# Main execution
echo "=== Hugo IndexNow Automation ==="

# Check if sitemap exists
if [ ! -f "public/sitemap.xml" ]; then
    echo "Error: sitemap.xml not found in public folder"
    echo "Make sure Hugo is configured to generate a sitemap"
    exit 1
fi

# Extract URLs from sitemap
echo "Extracting URLs from sitemap..."
urls=$(grep -oP '(?<=<loc>)[^<]+' public/sitemap.xml | grep -v "\.xml$" | head -10000)

if [ -z "$urls" ]; then
    echo "No URLs found in sitemap"
    exit 1
fi

url_count=$(echo "$urls" | wc -l)
echo "Found $url_count URLs to submit"

# Convert URLs to JSON array format
url_array=$(echo "$urls" | sed 's/.*/"&"/' | paste -sd ',' -)

# Create JSON payload
urls_json=$(cat << EOF
{
  "host": "$SITE_URL",
  "key": "$API_KEY",
  "keyLocation": "$KEY_LOCATION",
  "urlList": [$url_array]
}
EOF
)

# Submit URLs
submit_to_indexnow "$urls_json"
