import json
import os
from typing import Dict, List, Union

import numpy as np
import transformers
from flask import Flask, request, jsonify
from flask_cors import CORS

from prepare_haystack_4_pipeline import build_pipeline, clean_text

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Define the app
app = Flask(__name__)

# Set CORS policies
CORS(app)

haystack_pipeline = build_pipeline(run_index = True)

@app.route("/query", methods=["GET"])
def qa():
  print("query called")
  result = {}
  if request.args.get("query"):
    query = request.args.get("query")

    result = haystack_pipeline.run(
        query=clean_text(query),
        params={
          "BMRetriever": {"top_k": 3},
          "EMRetriever": {"top_k": 3}
        })
  else:
    return {"error": "Couldn't process your request"}, 422

  print("QUERY", query)

  print("ANSWERS:")
  for a in result.get('answers', []):
    print('title', a.meta['title'])
    print('score', a.score)
    print('entities', a.meta['entities'])
    print('content_id', a.meta['content_id'])
    print('context', a.context)
    print()

  print("DOCUMENTS:")
  for d in result.get('documents', []):
    print(d.meta['title'])
    print(d.meta['content_id'])
    print(d.content)
    print(d.score)
    print()

  final_result = []
  for d in result.get('documents', []):

    result = {
      'title': d.meta['title'],
      'subject': d.meta['subject'],
      'section_entities': d.meta['entities'],
      'content_id': d.meta['content_id'],
      'score': min(100.0, 50.0 + d.score * 55.0)
    }
    final_result.append(result)

  return jsonify(final_result)


if __name__ == '__main__':
  app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
