import argparse
import shlex
import json
from fastapi import FastAPI, Request
import uvicorn
from pathlib import Path

import torch
import transformers

import src.util
from src.options import Options
import src.data as data_module
import src.model as model_module


def clean_decoded(ans):
    ans = ans.replace("{{", " {{").replace("\n", ' ').replace("\r", "").replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip()
    return " ".join(ans.split())


def create_app(opt):
    app = FastAPI()

    # load tokenizer
    if 'codet5' in opt.tokenizer_name:
        tokenizer = transformers.RobertaTokenizer.from_pretrained(opt.tokenizer_name)
    else:
        tokenizer = transformers.T5Tokenizer.from_pretrained(opt.tokenizer_name)

    # load model
    model_class = model_module.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu'))
    model = model.to(device)
    model.eval()

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/answer')
    async def answer(request: Request):
        payload = await request.json()
        # accept either a single item or list
        items = payload if isinstance(payload, list) else [payload]

        # prepare text_passages for encode_passages: list of lists
        text_passages = []
        ids = []
        for ex in items:
            q = ex.get('question', '')
            ctxs = ex.get('ctxs', [])[:opt.n_context]
            # format passages as in Dataset: title_prefix + title + passage_prefix + text
            passages = []
            f = opt.title_prefix + " {} " + opt.passage_prefix + " {}"
            if len(ctxs) == 0:
                passages = [q]
            else:
                for c in ctxs:
                    title = c.get('title', '')
                    text = c.get('text', '')
                    passages.append(f.format(title, text))
            # prepend question to each passage when encoding (same as Collator.append_question)
            passages_with_q = [q + " " + p for p in passages]
            text_passages.append(passages_with_q)
            ids.append(ex.get('id', None))

        # encode
        passage_ids, passage_masks = data_module.encode_passages(text_passages, tokenizer, opt.text_maxlength)

        # generate
        with torch.no_grad():
            passage_ids = passage_ids.to(device)
            passage_masks = passage_masks.to(device)

            outputs = model.generate(
                input_ids=passage_ids,
                attention_mask=passage_masks,
                max_length=opt.max_length,
                lenpen=opt.lenpen,
                num_beams=opt.num_beams,
                temperature=opt.temperature,
                top_p=opt.top_p,
                num_return_sequences=opt.num_return_sequences,
            )

        results = []
        if opt.num_return_sequences == 1:
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=False)
                ans = clean_decoded(ans)
                results.append({'question_id': ids[k], 'clean_code': ans})
        else:
            # reshape outputs (batch, num_return_sequences, seq_len)
            outputs = outputs.view(-1, opt.num_return_sequences, outputs.size(-1))
            for k, o in enumerate(outputs):
                ans_list = []
                for oj in o:
                    ans = tokenizer.decode(oj, skip_special_tokens=False)
                    ans_list.append(clean_decoded(ans))
                results.append({'question_id': ids[k], 'clean_code': ans_list})

        return results

    return app


def parse_args(in_program_call=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--device', default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--n_context', type=int, default=10)
    parser.add_argument('--text_maxlength', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--lenpen', type=float, default=1.0)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--title_prefix', type=str, default='title:')
    parser.add_argument('--passage_prefix', type=str, default='context:')
    args = parser.parse_args() if in_program_call is None else parser.parse_args(shlex.split(in_program_call))
    return args


if __name__ == '__main__':
    opt = parse_args()
    app = create_app(opt)
    uvicorn.run(app, host='0.0.0.0', port=opt.port)
