Quick usage:

Start the API:
python run_api.py --model_name <MODEL> --target_file <TARGET_TXT> --target_id_file <TARGET_ID> --target_embed_save_file <TARGET_EMB_PREFIX>
Example curl:
curl -X POST -H "Content-Type: application/json" -d '{"question":"Your question here"}' http://localhost:5000/retrieve
Notes:

The script expects a NumPy file at <TARGET_EMB_PREFIX>.npy, a text file of target sentences (--target_file) and a corresponding id file (--target_id_file).
Response is a JSON array with objects matching the example_fid_data.json structure (fields: id, question, target, answers, ctxs).

How to run (example):

Start the reader API:
python run_reader_api.py --model_path <CHECKPOINT_PATH> --tokenizer_name <TOKENIZER> --port 5001
Send the JSON returned by run_api.py (or example_fid_data.json) to get answers:
curl -X POST -H "Content-Type: application/json" -d @example_fid_data.json http://localhost:5001/answer
Next steps: I can run a quick smoke test (if you provide a model checkpoint and embeddings), or adjust output formatting to match any specific schema you want.


Example run:

./start_and_query.sh --retriever-args "--model_name MODEL --target_file data.txt --target_id_file data.id --target_embed_save_file embeddings" --reader-args "--model_path CHECKPOINT --tokenizer_name TOKENIZER" --question "Your question"