import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')

    query = tokenizer.apply_chat_template([
        {"role": "system", "content": f"Generate your reasoning chain with no more than 500 tokens"}, {"role": "user", "content": "Solve the equation: $\\frac{47}{9}+\\frac{3}{2} x=\\frac{5}{3}\\left(\\frac{5}{2} x+1\\right)$"}, {"role": "assistant", "content": f"<think>\n\ndfdsfgdsgsd blah blah\n\n</think>\n$\\boxed{{\\frac{4}{3}$}}"}
    ], add_generation_prompt=False, tokenize=False)
    print(query)

    url = "http://127.0.0.1:5000/get_reward"
    score = remote_rm_fn(url, [query])
    print(score)
