#!/usr/bin/env bash

# This script submits Hugo site URLs to IndexNow and Bing APIs

# Get environment variables (set by GitHub Actions)
SITE_URL="${SITE_URL:-https://learn-software.com}"
API_KEY="${API_KEY:-53f1811377874f608f161d768a9c0b78}"
KEY_LOCATION="$SITE_URL/$API_KEY.txt"
SITEMAP_INDEX="public/sitemap.xml"

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

# Early validation
if [ ! -f "$SITEMAP_INDEX" ]; then
  echo "Error: sitemap index not found at $SITEMAP_INDEX" >&2
  exit 1
fi

# Extract sitemap paths
sitemap_paths=$(grep -oE '<loc>[^<]+' "$SITEMAP_INDEX" | sed 's/<loc>//')
if [ -z "$sitemap_paths" ]; then
  echo "No sitemap entries found in sitemap index" >&2
  exit 1
else
  printf "Found %d Sitemaps to scan.\n" "$(printf "%s\n" "$sitemap_paths" | wc -l)"
  echo $sitemap_paths
fi

url_list=""
while IFS= read -r sitemap_url; do
  relative=$(echo "$sitemap_url" | sed -E 's~https?://[^/]+/~~')
  local_path="public/$relative"
  if [ ! -f "$local_path" ]; then
    echo "Warning: missing $local_path, skipping…" >&2
    continue
  fi
  page_urls=$(grep -oE '<loc>[^<]+' "$local_path" | sed 's/<loc>//')
  url_list="${url_list}"$'\n'"${page_urls}"
done <<< "$sitemap_paths"

# Sanitize: remove blanks and duplicates
url_list=$(printf "%s\n" "$url_list" | sed '/^\s*$/d' | sort -u)
if [ -z "$url_list" ]; then
  echo "No URLs found in any sitemap" >&2
  exit 1
fi

printf "Found %d URLs to submit.\n" "$(printf "%s\n" "$url_list" | wc -l)"

# Build a comma-separated list
url_array=""
first=true
while IFS= read -r url; do
  # Skip empty lines
  [ -z "$url" ] && continue

  # Escape double quotes and backslashes
  esc_url=$(printf '%s' "$url" | sed 's/\\/\\\\/g; s/"/\\"/g')

  if $first; then
    url_array="\"$esc_url\""
    first=false
  else
    url_array="$url_array, \"$esc_url\""
  fi
done <<< "$url_list"

# Create JSON payload
urls_json=$(cat << EOF
{
  "host": "$SITE_URL",
  "key": "$API_KEY",
  "keyLocation": "$KEY_LOCATION",
  "urlList": [ $url_array ]
}
EOF
)

echo $urls_json

# Submit URLs
# submit_to_indexnow "$urls_json"
