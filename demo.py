import json
import os
from pprint import pprint
import re
from tqdm import tqdm
import random

import uuid
import openai
from openai import OpenAI
import tiktoken
import json
import numpy as np
import requests
from scipy import sparse
#from rank_bm25 import BM25Okapi
#import jieba
from http import HTTPStatus


from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
import time
from tqdm import tqdm

logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip") # 日志信息

MODEL_NAME = 'glm-4-9b-lora' # 模型名称

# https://github.com/THUDM/GLM-4
#
# https://huggingface.co/THUDM/glm-4-9b-chat

# VLLM环境：https://docs.vllm.ai/en/latest/
#%%
def call_qwen_api(MODEL_NAME, query): # API接口
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
    )

    completion = client.chat.completions.create(
      model=MODEL_NAME,
      messages=[
        {"role": "system",
        "content": '你是一位自然语言处理专家，擅长处理文本分类和情感分析任务。'},
        {"role": "user", "content": query}
      ], timeout=30
    )
    return completion.choices[0].message.content
#%%

print(call_qwen_api(MODEL_NAME, query="你是谁"))