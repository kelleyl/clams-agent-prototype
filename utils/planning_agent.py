"""
Enhanced planning agent for CLAMS pipelines.
Generates structured pipeline plans through conversational interaction.
"""

from typing import List, Dict, Any, Optional
import json
import logging
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .pipeline_execution import PipelinePlan, ToolStep
from .clams_tools import CLAMSToolbox
from .config import ConfigManager

logger = logging.getLogger(__name__)

class PipelinePlanOutput(BaseModel):
    """Pydantic model for structured pipeline plan output."""
    steps: List[Dict[str, Any]] = Field(description="List of pipeline steps with tool_name, parameters, config, and reasoning")
    reasoning: str = Field(description="Overall reasoning for the pipeline design")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    input_types: List[str] = Field(description="Expected input types (e.g., ['VideoDocument'])")
    output_types: List[str] = Field(description="Expected output types (e.g., ['TextDocument', 'Alignment'])")
    estimated_time: int = Field(description="Estimated total execution time in seconds")

class CLAMSPlanningAgent:
    """Enhanced planning agent that generates structured pipeline plans."""
    
    def __init__(self, config_manager: ConfigManager = None):
        """Initialize the planning agent."""
        self.config_manager = config_manager or ConfigManager()
        self.llm_config = self.config_manager.get_config().llm
        
        # Initialize LLM
        if self.llm_config.provider == "ollama":
            self.llm = ChatOllama(
                model=self.llm_config.model_name,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                top_p=self.llm_config.top_p
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.llm_config.model_name,
                temperature=self.llm_config.temperature
            )
        
        # Initialize tools and metadata
        self.toolbox = CLAMSToolbox()
        self.tool_metadata = self._initialize_tool_metadata()
        
        # Set up output parser
        self.output_parser = PydanticOutputParser(pydantic_object=PipelinePlanOutput)
    
    def _initialize_tool_metadata(self) -> Dict[str, Any]:
        """Initialize tool metadata for pipeline planning."""
        metadata = {}
        
        for tool_name, clams_tool in self.toolbox.get_tools().items():
            app_info = clams_tool.app_metadata
            tool_metadata = app_info.get('metadata', {})
            
            # Extract and clean input/output types
            input_types = self._extract_types(tool_metadata.get('input', []))
            output_types = self._extract_types(tool_metadata.get('output', []))
            
            metadata[tool_name] = {
                'description': tool_metadata.get('description', ''),
                'input_types': input_types,
                'output_types': output_types,
                'parameters': tool_metadata.get('parameters', []),
                'app_version': tool_metadata.get('app_version', 'unknown')
            }
        
        return metadata
    
    def _extract_types(self, type_list: List[Dict[str, Any]]) -> List[str]:
        """Extract clean type names from MMIF type URIs."""
        types = []
        for type_info in type_list:
            if isinstance(type_info, dict) and '@type' in type_info:
                type_uri = type_info['@type']
                # Extract clean type name
                type_name = type_uri.split('/')[-1]
                # Remove version numbers
                clean_name = type_name.split('v')[0] if 'v' in type_name else type_name
                if clean_name:
                    types.append(clean_name)
        return types
    
    def _create_planning_prompt(self, user_query: str) -> str:
        """Create a comprehensive planning prompt."""
        tool_descriptions = self._get_tool_descriptions()
        
        format_instructions = self.output_parser.get_format_instructions()
        
        return f"""You are a CLAMS (Computational Language and Audiovisual Multimedia Systems) pipeline expert.

Your task is to analyze the user's request and create a structured pipeline plan using available CLAMS tools.

User Request: {user_query}

Available CLAMS Tools:
{tool_descriptions}

Pipeline Planning Rules:
1. Consider tool input/output type compatibility
2. Video processing typically starts with VideoDocument input
3. OCR tools need TimeFrame annotations or images
4. Speech recognition tools need audio streams
5. Text analysis tools need transcribed text
6. Chain tools logically (output of one becomes input of next)

For each step, specify:
- tool_name: Exact tool name from the available tools
- parameters: Key-value pairs for tool configuration (if needed)
- config: Configuration file name (if applicable, e.g., "default.yaml")
- reasoning: Why this tool is needed and how it fits in the pipeline

Confidence Scoring:
- 1.0: Perfect match, all tools available and well-suited
- 0.8: Good match, minor limitations or assumptions
- 0.6: Reasonable match, some tools may not be optimal
- 0.4: Partial match, significant limitations
- 0.2: Poor match, major issues or missing capabilities

{format_instructions}

Create a structured pipeline plan that addresses the user's request:"""
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        descriptions = []
        
        for tool_name, metadata in self.tool_metadata.items():
            input_types = ", ".join(metadata['input_types']) if metadata['input_types'] else "Any"
            output_types = ", ".join(metadata['output_types']) if metadata['output_types'] else "Various"
            
            descriptions.append(
                f"- {tool_name}: {metadata['description']}\n"
                f"  Input: {input_types} | Output: {output_types}"
            )
        
        return "\n".join(descriptions)
    
    async def suggest_pipeline(self, user_query: str) -> PipelinePlan:
        """
        Generate a structured pipeline plan based on user query.
        
        Args:
            user_query: User's description of what they want to accomplish
            
        Returns:
            PipelinePlan: Structured pipeline plan
        """
        logger.info(f"Planning pipeline for query: {user_query}")
        
        try:
            # Create planning prompt
            prompt = self._create_planning_prompt(user_query)
            
            # Get LLM response
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            
            # Parse structured output
            try:
                parsed_output = self.output_parser.parse(response.content)
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {e}")
                # Fallback to manual parsing
                parsed_output = self._fallback_parse(response.content, user_query)
            
            # Convert to PipelinePlan
            steps = []
            for step_data in parsed_output.steps:
                step = ToolStep(
                    tool_name=step_data.get('tool_name', ''),
                    parameters=step_data.get('parameters', {}),
                    config=step_data.get('config'),
                    reasoning=step_data.get('reasoning', ''),
                    estimated_time=step_data.get('estimated_time', 30)
                )
                steps.append(step)
            
            plan = PipelinePlan(
                steps=steps,
                reasoning=parsed_output.reasoning,
                estimated_total_time=parsed_output.estimated_time,
                confidence=parsed_output.confidence,
                input_types=parsed_output.input_types,
                output_types=parsed_output.output_types
            )
            
            logger.info(f"Generated plan with {len(steps)} steps, confidence: {plan.confidence}")
            return plan
            
        except Exception as e:
            logger.error(f"Pipeline planning failed: {e}")
            # Return a fallback plan
            return self._create_fallback_plan(user_query, str(e))
    
    def _fallback_parse(self, response_content: str, user_query: str) -> PipelinePlanOutput:
        """Fallback parsing when structured output fails."""
        logger.info("Using fallback parsing for pipeline plan")
        
        # Simple heuristic-based planning
        steps = []
        confidence = 0.5
        reasoning = f"Fallback plan for: {user_query}"
        
        # Common patterns
        if any(word in user_query.lower() for word in ['speech', 'spoken', 'audio', 'transcribe']):
            steps.append({
                'tool_name': 'whisper-wrapper',
                'parameters': {},
                'config': None,
                'reasoning': 'Extract spoken text from audio'
            })
        
        if any(word in user_query.lower() for word in ['text', 'ocr', 'read', 'visual']):
            steps.append({
                'tool_name': 'easyocr-wrapper',
                'parameters': {},
                'config': None,
                'reasoning': 'Extract text from video frames'
            })
        
        if any(word in user_query.lower() for word in ['scene', 'shot', 'segment']):
            steps.append({
                'tool_name': 'swt-detection',
                'parameters': {},
                'config': None,
                'reasoning': 'Detect scenes and segments'
            })
        
        if not steps:
            # Default pipeline
            steps.append({
                'tool_name': 'whisper-wrapper',
                'parameters': {},
                'config': None,
                'reasoning': 'General audio analysis'
            })
        
        return PipelinePlanOutput(
            steps=steps,
            reasoning=reasoning,
            confidence=confidence,
            input_types=['VideoDocument'],
            output_types=['TextDocument'],
            estimated_time=len(steps) * 60
        )
    
    def _create_fallback_plan(self, user_query: str, error: str) -> PipelinePlan:
        """Create a minimal fallback plan when planning fails."""
        step = ToolStep(
            tool_name='whisper-wrapper',
            parameters={},
            config=None,
            reasoning=f'Fallback plan due to planning error: {error}',
            estimated_time=60
        )
        
        return PipelinePlan(
            steps=[step],
            reasoning=f"Fallback plan for query: {user_query}. Error: {error}",
            estimated_total_time=60,
            confidence=0.1,
            input_types=['VideoDocument'],
            output_types=['TextDocument']
        )
    
    async def conversational_planning(self, user_query: str) -> str:
        """
        Generate a conversational explanation of the pipeline plan.
        
        Args:
            user_query: User's request
            
        Returns:
            Conversational explanation of the suggested pipeline
        """
        plan = await self.suggest_pipeline(user_query)
        
        explanation = f"""Based on your request: "{user_query}"

I suggest the following pipeline approach:

**Pipeline Overview:**
{plan.reasoning}

**Execution Steps:**
"""
        
        for i, step in enumerate(plan.steps, 1):
            explanation += f"{i}. **{step.tool_name}**: {step.reasoning}\n"
            if step.parameters:
                explanation += f"   Parameters: {step.parameters}\n"
            if step.config:
                explanation += f"   Configuration: {step.config}\n"
            explanation += "\n"
        
        explanation += f"""
**Estimated Time:** {plan.estimated_total_time // 60} minutes {plan.estimated_total_time % 60} seconds
**Confidence:** {plan.confidence:.1%}

Would you like me to execute this pipeline, or would you prefer to modify any steps?
"""
        
        return explanation