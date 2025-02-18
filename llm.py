from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from typing import List, Dict


class LLMHandler:
    def __init__(self, api_key: str = None):
        """
        Initialize with Groq API token
        Get it from: https://console.groq.com/
        """
        # Load environment variables
        load_dotenv()

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Please provide a Groq API key via GROQ_API_KEY environment variable"
            )

        # Debug print (first 8 chars of API key)
        print(f"Initializing Groq with API key starting with: {self.api_key[:8]}...")

        # Initialize Groq LLM
        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=4096,
            )
            print("Groq LLM initialized successfully!")
        except Exception as e:
            raise Exception(f"Failed to initialize Groq LLM: {str(e)}")

    def generate_summary(self, context: str, max_length: int = 300) -> str:
        """
        Generate a summary using Groq's LLM
        """
        messages = [
            SystemMessage(
                content="You are an expert at summarizing academic papers. Provide clear and concise summaries."
            ),
            HumanMessage(
                content=f"Summarize the key points of this paper briefly:\n\nContext: {context}"
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")

    def generate_refined_summary(self, text: str, similar_papers: List[Dict]) -> str:
        """Generate a summary with context from similar papers"""
        try:
            # Create context from similar papers
            similar_context = "\n\n".join(
                [
                    f"Related Paper: {p['title']}\nSummary: {p['generated_summary']}"
                    for p in similar_papers
                ]
            )

            # Generate enhanced summary
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="""You are an expert at analyzing academic papers and providing insights.
                        Use the context from similar papers to provide a more informed summary."""
                    ),
                    HumanMessage(
                        content=f"""Summarize this paper and relate it to similar research:

                        Paper to Summarize:
                        {text}

                        Similar Research Context:
                        {similar_context}

                        Provide:
                        1. Main points of the current paper
                        2. How it relates to similar research
                        3. Key differences or advancements
                        """
                    ),
                ]
            )
            return response.content.strip()
        except Exception as e:
            raise Exception(f"Refined summary generation failed: {str(e)}")
