from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from transformers import ReactCodeAgent, HfApiEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    """Stores context for pipeline construction."""
    task_description: str
    current_step: int = 1
    selected_tools: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.selected_tools = self.selected_tools or []
    
    def add_selected_tool(self, tool_name: str, tool_info: Dict[str, Any]):
        """Add a selected tool to the pipeline."""
        self.selected_tools.append({
            "step": self.current_step,
            "name": tool_name,
            "info": tool_info
        })
        self.current_step += 1
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state as a formatted string."""
        if not self.selected_tools:
            return "No tools selected yet."
            
        state = "Current pipeline:\n"
        for tool in self.selected_tools:
            state += f"Step {tool['step']}: {tool['name']}\n"
        return state

class PipelineAgentChat:
    """
    Agent-based chat interface for CLAMS pipeline generation using Hugging Face's Transformers Agents.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-70b-chat-hf"):
        """
        Initialize the pipeline agent chat.
        
        Args:
            model_name: Name of the model to use for the agent
        """
        # Initialize the LLM engine
        self.llm_engine = HfApiEngine(model=model_name)
        
        # Create the agent with custom system prompt
        system_prompt = """You are a CLAMS pipeline assistant helping users create optimal pipelines for video analysis tasks.
Your goal is to help users construct pipelines by selecting and connecting appropriate CLAMS tools.

When suggesting tools:
1. Only suggest tools that are directly relevant to the current step
2. Explain why each tool is needed and what it contributes
3. Consider the input/output requirements of each tool
4. Aim for efficiency - avoid unnecessary tools
5. For video text analysis tasks, focus on video/image processing tools

DO NOT:
1. Generate or simulate user messages
2. Suggest tools that don't exist in the metadata
3. Add unnecessary steps to the pipeline

Format your responses clearly and concisely, focusing on:
1. What tools are available for the current step
2. Why each tool might be appropriate
3. What inputs they require and outputs they produce
4. Any important parameters to consider

Remember: You are helping to construct a real pipeline that will be executed, so accuracy and practicality are essential."""

        self.agent = ReactCodeAgent(
            tools=[],  # We'll add CLAMS tools later
            llm_engine=self.llm_engine,
            system_prompt=system_prompt,
            add_base_tools=True  # Include basic tools like text processing
        )
        
    def chat_response(self, context: ChatContext, user_input: str) -> str:
        """
        Generate a response in the chat conversation using the agent.
        
        Args:
            context: Current chat context
            user_input: User's message
            
        Returns:
            Assistant's response
        """
        try:
            # Create the full prompt with context
            full_prompt = f"""Task: {context.task_description}

Current pipeline state:
{context.get_pipeline_state()}

User message: {user_input}

Please suggest appropriate tools for the next step in the pipeline, considering the current state and user's input."""

            # Run the agent
            response = self.agent.run(full_prompt)
            
            # Extract and process the response
            if isinstance(response, str):
                return response
            else:
                # If the agent returned multiple outputs, combine them
                return "\n".join(str(item) for item in response if item)
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"Error generating response: {str(e)}"
            
    def interactive_pipeline_design(self, task_description: str) -> Optional[str]:
        """
        Start an interactive chat session to design a pipeline.
        
        Args:
            task_description: Initial task description
            
        Returns:
            Generated pipeline description if successful, None otherwise
        """
        context = ChatContext(task_description=task_description)
        
        print("\nWelcome to the CLAMS Pipeline Designer!")
        print("Describe your task and I'll help you create an optimal pipeline.")
        print("Type 'done' when you're ready to generate the pipeline.")
        print("Type 'quit' to exit.\n")
        
        # Initial response
        initial_response = self.chat_response(
            context,
            f"I need to create a pipeline for this task: {task_description}"
        )
        print(f"\nAssistant: {initial_response}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    return None
                    
                if user_input.lower() == 'done':
                    # Generate final pipeline
                    final_prompt = f"""Based on our discussion about the task:
{task_description}

Current pipeline state:
{context.get_pipeline_state()}

Please generate a complete pipeline description following this format:

PIPELINE:
1. First tool name
   - Purpose: Why this tool is needed
   - Inputs: What inputs it uses and their source
   - Outputs: What outputs it produces
   - Parameters: Any specific parameters to set
2. Second tool name
   ...etc.

EXPLANATION:
A brief explanation of how the pipeline accomplishes the task and why this order is necessary."""

                    pipeline = self.agent.run(final_prompt)
                    return pipeline
                
                # Get assistant's response
                response = self.chat_response(context, user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue 