import os

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.azure_openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

df = pd.DataFrame(
    {
        "country": [
            "United States",
            "United Kingdom",
            "France",
            "Germany",
            "Italy",
            "Spain",
            "Canada",
            "Australia",
            "Japan",
            "China",
        ],
        "gdp": [
            19294482071552,
            2891615567872,
            2411255037952,
            3435817336832,
            1745433788416,
            1181205135360,
            1607402389504,
            1490967855104,
            4380756541440,
            14631844184064,
        ],
    }
)

# Instantiate a LLM
# from pandasai.llm import OpenAI
# llm = OpenAI(api_token="YOUR_API_TOKEN")  # Get API token from https://platform.openai.com/account/api-keys
llm = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

df = SmartDataframe(df, config={"llm": llm})
result = df.chat("Which are the countries with GDP greater than 3000000000000?")
print(result)
