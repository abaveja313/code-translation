import base64
from typing import List
import tempfile
from prefect import task
from loguru import logger


@task()
def evaluate(java_code: str, inputs: List[str], expected: List[str]):
   java_script_json = {
      "java_program": base64.b64encode(bytes(java_code, 'utf-8')),
      "test_cases": [
         {
            "test_case_name":
         }
      ]
      "inputs": inputs,
      "outputs": expected
   }
