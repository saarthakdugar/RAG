import logging
from openai import AsyncAzureOpenAI
from .. import config

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        if not all([
            config.AZURE_OPENAI_ENDPOINT, 
            config.AZURE_OPENAI_API_KEY, 
            config.AZURE_OPENAI_DEPLOYMENT_NAME, 
            config.AZURE_OPENAI_API_VERSION
        ]):
            logger.error("Azure OpenAI credentials not fully configured")
            raise ValueError("Azure OpenAI credentials not fully configured. Please check your .env file.")

        self.client = AsyncAzureOpenAI(
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        self.deployment_name = config.AZURE_OPENAI_DEPLOYMENT_NAME
        logger.info(f"LLM Service initialized with model: {self.deployment_name}")

    async def generate_response(self, messages: list[dict]) -> str:
        logger.info(f"Generating response using {self.deployment_name}")
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.deployment_name,
            )
            content = response.choices[0].message.content
            logger.info(f"Response received ({len(content) if content else 0} chars)")
            return content or "" 
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while trying to generate a response."
 