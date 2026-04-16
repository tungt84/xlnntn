import argparse
import json
import shlex
import uuid
import os

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import numpy as np
import faiss
import torch

import run_inference as run_inference_module


def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f]


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def pad_single(sentence, tokenizer):
    sent_features = tokenizer([
        sentence,
    ], add_special_tokens=True, max_length=tokenizer.model_max_length, truncation=True)
    arr = sent_features['input_ids']
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=torch.long) * tokenizer.pad_token_id
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=torch.long)
        mask[i, : lens[i]] = 1
    return {'input_ids': padded, 'attention_mask': mask, 'lengths': lens}


def create_app(args):
    app = FastAPI()

    # load target texts and ids
    target_texts = load_lines(args.target_file)
    target_ids = load_lines(args.target_id_file)

    # load embeddings
    emb = np.load(args.target_embed_save_file + '.npy')
    assert emb.shape[0] == len(target_texts)

    index = build_index(emb)

    # provide module-level args expected by run_inference internals
    run_inference_module.args = args
    retriever = run_inference_module.CodeT5Retriever(args)
    retriever.prepare_model()

    device = retriever.device

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/retrieve')
    async def retrieve(request: Request):
        data = await request.json()
        question = data.get('question') or data.get('q')
        if not question:
            raise HTTPException(status_code=400, detail='missing question field')

        padded = pad_single(question, retriever.tokenizer)
        for k in padded:
            if isinstance(padded[k], torch.Tensor):
                padded[k] = padded[k].to(device)

        with torch.no_grad():
            q_embed = retriever.model.get_pooling_embedding(**padded, normalize=args.normalize_embed).detach().cpu().numpy()

        D, I = index.search(q_embed, args.top_k)

        top_indices = I[0].tolist()
        top_scores = D[0].tolist()

        # Build a response following the example_fid_data.json structure
        item = {}
        item['id'] = str(uuid.uuid4())
        item['question'] = question
        # primary target: top-1 text
        primary_idx = top_indices[0] if top_indices else None
        item['target'] =  ''
        # answers: include best text as single-item list
        item['answers'] =  []

        # ctxs: include retrieved top-k entries with scores
        ctxs = []
        for idx, score in zip(top_indices, top_scores):
            ctxs.append({
                'title': target_ids[idx],
                'text': target_texts[idx],
                'man_id': target_ids[idx]
            })
        item['ctxs'] = ctxs

        return [item]

    return app


def parse_args(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--target_file', type=str, required=True)
    parser.add_argument('--target_id_file', type=str, required=True)
    parser.add_argument('--target_embed_save_file', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--normalize_embed', action='store_true')
    parser.add_argument('--log_level', type=str, default='verbose')
    parser.add_argument('--sim_func', type=str, default='cls_distance.cosine')
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--pooler', choices=('cls', 'cls_before_pooler'), default='cls')
    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    return args


if __name__ == '__main__':
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host='0.0.0.0', port=args.port)
