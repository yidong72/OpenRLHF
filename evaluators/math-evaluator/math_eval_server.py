import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger
from parser import extract_answer

logger = init_logger(__name__)


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = extract_answer

        # load the ground truths
        self.gt_ans = {}
        with open(args.dataset, 'r') as f:
            for l in f.readlines():
                l = json.loads(l.strip())
                conv = l['input']
                user_content = conv[0]['content'].strip() if conv[0]['role'] == 'user' else conv[1]['content'].strip()
                self.gt_ans[user_content] = [l['gt_ans'], l['metadata']['dataset']]

    def extract_generation(self, text):
        try:
            return text.split('</think>')[-1].replace('<|eot_id|>', '')
        except:
            return text

    def get_reward(self, queries):
        scores = []
        for i in range(len(queries)):
            query = self.extract_generation(queries[i])
            prompt = queries[i].split('<|start_header_id|>user<|end_header_id|>')[-1].split('<|eot_id|>')[0].strip()
            query_data = self.gt_ans[prompt]
            extracted_answer = self.reward_model(query, query_data[1], use_last_number=False)

            logger.info(f"prompt and answer: {prompt} | {extracted_answer}")
            logger.info(f"prompt data: {query_data}")

            # reward function
            score = 1 if extracted_answer == query_data[0] else 0
            scores.append(score)

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    parser.add_argument("--dataset", type=str, default="0.0.0.0", help="Dataset including gt_ans")

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")