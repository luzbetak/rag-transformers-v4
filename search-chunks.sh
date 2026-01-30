#!/bin/bash

# search-chunks.sh
# Search chunks collection by text pattern

if [ -z "$1" ]; then
  echo "Usage: ./search-chunks.sh <search_term>"
  echo "Example: ./search-chunks.sh vitamin"
  exit 1
fi

SEARCH_TERM="$1"

mongosh << EOF
use books
db.chunks.find({text: /$SEARCH_TERM/i}).limit(1).pretty()
EOF
