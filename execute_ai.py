# Imports various components from Langchain's agent system that help define and execute AI agents.
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser

# Used to create custom string-based prompt templates.
from langchain.prompts import StringPromptTemplate

# OpenAI: interface for OpenAI's LLMs.
# SerpAPIWrapper: enables Google search via SerpAPI.
# LLMChain: links an LLM with a prompt template.
from langchain import OpenAI, SerpAPIWrapper, LLMChain

# Python type hints used for function signatures and static checks.
from typing import List, Union

# Schema definitions for agent actions (intermediate steps) and final results.
from langchain.schema import AgentAction, AgentFinish

# File tools to read from and write to files during the agent's workflow.
from langchain.tools.file_management import WriteFileTool
 
# openai: direct SDK access.
# re: regex operations.
# TemporaryDirectory: create temp file paths for intermediate work.
from langchain.tools.file_management.read import ReadFileTool
import openai
import re
from tempfile import TemporaryDirectory

# Enables interaction with Zapier’s Natural Language Actions API.
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit

# Used to create and configure an AI agent.
from langchain.agents import initialize_agent

# Standard library to interact with the file system and environment variables.
import os

# Provides available agent types (e.g., ZERO_SHOT_REACT_DESCRIPTION).
from langchain.agents import AgentType

# Utility to create structured prompt templates.
from langchain import PromptTemplate

# Defines a function to initialize and return a Zapier-based AI agent.
def call_agent():
    
    # Creates a wrapper to use Zapier’s natural language API for tasks like messaging, calendar booking, etc.
    zapier = ZapierNLAWrapper()

    # Creates a toolkit of available Zapier actions (like creating events or sending messages).
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    # Initializes the LLM with deterministic behavior (temperature=0 means no randomness).
    llm = OpenAI(temperature=0)

    # Creates a Zero-shot ReAct agent, which decides tool usage based on descriptions.
    # verbose=True enables detailed logging for debugging or explanation.
    agent = initialize_agent(toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Returns the configured agent instance.
    return agent

# Function that returns an LLM chain for generating friendly answers.
def answer_the_call():

    # Instruction prompt to the LLM that it's acting as a receptionist.
    # Provides few-shot examples of caller inputs and expected outputs.
    # Ends with placeholders ({CALLER}, {INFO}) for dynamic input.
    template = """
    You are an AI receptionist to a paper company, receiving calls from various people. 
    Your fellow AI recptionist has executed the follwing tasks according to the caller's request. 
    Tell this informtion to the caller in a friendly manner.
    Here are some examples:
    ======
    CALLER: Tell Michael that he missed yesterday's lunch. 
    INFO: The message "Michael, you missed yesterday's lunch" has been successfully appended to the Messages document
    ANSWER: Your message has been successfully noted. Is there anything else I can help you with?
    
    CALLER: Schedule an appointment with Jim for 2pm on the 26th of May
    INFO: The calendar invite has been successfully created.
    ANSWER: Your meeting has been successfully scheduled. Is there anything else I can help you with?

    CALLER: Is Toby free for 2pm on the 23rd of May?
    INFO: The calendar shows an appointment for 23rd of May.
    ANSWER: I am sorry, he is not free at that particular slot. Is there anything else I can help you with?

    CALLER: No, thank you. 
    INFO: Not enough information provided in the instruction, missing <param>.
    ANSWER: Thank you for calling, bye!

    ========
    {CALLER}
    ========
    {INFO}
    =====
    ANSWER: 
    """
    # Again, initializes an OpenAI model for generating responses.
    llm = OpenAI(temperature=0)

    # Builds a prompt template that takes the INFO (tool result or message) as input.
    receptionist_prompt = PromptTemplate(template=template, input_variables=["INFO"])

    # Combines the LLM and the prompt into a chain that can be executed with dynamic input.
    llm_answer_chain = LLMChain(llm=llm, prompt=receptionist_prompt)

    # Returns the LLM chain for later use in generating final spoken responses.
    return llm_answer_chain
