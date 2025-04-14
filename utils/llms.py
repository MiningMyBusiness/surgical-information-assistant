from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

def init_llm(llm_name: str='azure-gpt4', llm_temperature: float=0.7):
    if 'azure' in llm_name:
        return init_llm_azure(llm_name, llm_temperature)
    elif 'ollama' in llm_name:
        return init_llm_ollama(llm_name, llm_temperature)
    else:
        return init_llm_together(llm_name, llm_temperature)

def init_llm_azure(llm_name: str='azure-gpt4', llm_temperature: float=0.7):
    """
    :param: None
    :return: the llm, LangChain LLM object
    :Description: Initializes the LLM endpoint that will be used by Lang Chain.
    """
    if llm_name == 'azure-gpt35':
        llm = AzureChatOpenAI(
            openai_api_version="2023-12-01-preview",
            azure_deployment='gpt-35-turbo-manuscript-gen',
            model_name="gpt-35-turbo",
            temperature=llm_temperature,
        )
        return llm
    elif llm_name == 'azure-gpt4':
        llm = AzureChatOpenAI(
            openai_api_version="2024-08-01-preview",
            azure_deployment='gpt-4o',
            model_name="gpt-4o",
            temperature=llm_temperature,
        )
        return llm
    elif llm_name == 'azure-o1':
        llm = O1Caller()
        return llm
    else:
        raise ValueError(f"Unsupported Azure LLM model: {llm_name}")
    

def init_llm_together(llm_name: str='together-llama33', llm_temperature: float=0.7):
    llm_name = llm_name.split('together-')[-1].lower()
    model_name = 'together_' + llm_name
    llm = ChatOpenAI(
        model = os.environ.get(model_name.upper()),
        base_url = "https://api.together.xyz/v1/",
        api_key = os.environ.get('TOGETHER_API_KEY'),
        temperature = llm_temperature,
    )
    return llm


class O1Caller:

    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        )

    def invoke(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": f"{user_prompt}",
                }
            ],
            model="o1-v2"
        )

        return response.choices[0].message
    

def init_llm_ollama(llm_name: str='ollama-llama3.1', llm_temperature: float=0.7):
    llm_name = llm_name.split('ollama-')[1]
    llm = ChatOllama(
        model = llm_name,
        base_url = os.environ.get('OLLAMA_ROUTE'),
        temperature = llm_temperature,
        num_ctx = 100000
    )
    return llm