from genai import Client, Credentials
from genai.schema import (
    TextTokenizationParameters,
    TextTokenizationReturnOptions,
)

from utils.load_config import LoadConfig

CONFIG = LoadConfig()


def count_num_tokens(text: str) -> int:
    credentials = Credentials(
        api_key=CONFIG.genai_api_key,
        api_endpoint="https://bam-api.res.ibm.com/v2/text/tokenization?version=2024-01-10",
    )
    client = Client(credentials=credentials)

    responses = list(
        client.text.tokenization.create(
            model_id=CONFIG.genai_model_id,
            input=[text],
            parameters=TextTokenizationParameters(
                return_options=TextTokenizationReturnOptions(
                    input_text=True,
                )
            ),
        )
    )
    return responses[0].results[0].token_count
