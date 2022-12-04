#!/usr/bin/env python3

import sys
from clip_retrieval.clip_client import ClipClient, Modality

# Check if an image file is specified as an argument
if len(sys.argv) != 2:
    print("Please specify an image file as an argument")
    sys.exit(1)

# Set up the clip-retrieval client with the LAION-5B dataset
laion5b_search_client = ClipClient(
    url="https://knn5.laion.ai/knn-service",
    indice_name="laion5B",
    num_images=400,
)

# Query the LAION-5B dataset with the specified image file
results = laion5b_search_client.query(image=sys.argv[1])

# Print the results in the terminal
print("Found {} similar images:".format(len(results)))
for result in results:
    print("- {} (Similarity: {})".format(result["url"], result["similarity"]))
