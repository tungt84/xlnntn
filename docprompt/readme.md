Quick usage:

Start the API:
python run_api.py --model_name <MODEL> --target_file <TARGET_TXT> --target_id_file <TARGET_ID> --target_embed_save_file <TARGET_EMB_PREFIX>
Example curl:
curl -X POST -H "Content-Type: application/json" -d '{"question":"Your question here"}' http://localhost:5000/retrieve
Notes:

The script expects a NumPy file at <TARGET_EMB_PREFIX>.npy, a text file of target sentences (--target_file) and a corresponding id file (--target_id_file).
Response is a JSON array with objects matching the example_fid_data.json structure (fields: id, question, target, answers, ctxs).

