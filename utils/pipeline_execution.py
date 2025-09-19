"""
Pipeline execution engine for CLAMS tools.
Handles direct tool execution with progress tracking.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .clams_tools import CLAMSToolbox

logger = logging.getLogger(__name__)

@dataclass
class ToolStep:
    """Represents a single tool execution step in a pipeline."""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    config: Optional[str] = None
    reasoning: str = ""
    estimated_time: int = 30  # seconds
    
@dataclass 
class PipelinePlan:
    """Structured pipeline plan with tools and execution metadata."""
    steps: List[ToolStep]
    reasoning: str
    estimated_total_time: int
    confidence: float
    input_types: List[str]
    output_types: List[str]
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class StepResult:
    """Result of executing a single pipeline step."""
    step: ToolStep
    success: bool
    output: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

@dataclass
class ExecutionResult:
    """Complete pipeline execution result."""
    plan: PipelinePlan
    step_results: List[StepResult]
    success: bool
    total_time: float
    final_output: str = ""
    error_summary: Optional[str] = None
    
@dataclass
class ExecutionProgress:
    """Progress update during pipeline execution."""
    current_step: int
    total_steps: int
    step_name: str
    status: str  # "starting", "running", "completed", "failed"
    message: str = ""
    percentage: float = 0.0

class CLAMSExecutionEngine:
    """Direct execution engine for CLAMS tool pipelines."""
    
    def __init__(self):
        """Initialize the execution engine."""
        self.toolbox = CLAMSToolbox()
        self.tools = self.toolbox.get_tools()
        
    async def execute_plan(self, 
                          plan: PipelinePlan, 
                          input_mmif: str) -> AsyncGenerator[ExecutionProgress, None]:
        """
        Execute a pipeline plan with progress streaming.
        
        Args:
            plan: The pipeline plan to execute
            input_mmif: Input MMIF data or file path
            
        Yields:
            ExecutionProgress updates during execution
        """
        logger.info(f"Starting execution of plan {plan.plan_id}")
        
        step_results = []
        current_mmif = input_mmif
        total_steps = len(plan.steps)
        start_time = datetime.now()
        
        try:
            for i, step in enumerate(plan.steps):
                # Progress update: starting step
                yield ExecutionProgress(
                    current_step=i + 1,
                    total_steps=total_steps,
                    step_name=step.tool_name,
                    status="starting",
                    message=f"Initializing {step.tool_name}",
                    percentage=(i / total_steps) * 100
                )
                
                # Execute the step
                step_start = datetime.now()
                yield ExecutionProgress(
                    current_step=i + 1,
                    total_steps=total_steps,
                    step_name=step.tool_name,
                    status="running",
                    message=f"Executing {step.tool_name}...",
                    percentage=((i + 0.5) / total_steps) * 100
                )
                
                try:
                    result = await self._execute_step(step, current_mmif)
                    step_end = datetime.now()
                    result.execution_time = (step_end - step_start).total_seconds()
                    result.end_time = step_end.isoformat()
                    
                    step_results.append(result)
                    
                    if result.success:
                        current_mmif = result.output
                        yield ExecutionProgress(
                            current_step=i + 1,
                            total_steps=total_steps,
                            step_name=step.tool_name,
                            status="completed",
                            message=f"✓ {step.tool_name} completed successfully",
                            percentage=((i + 1) / total_steps) * 100
                        )
                    else:
                        yield ExecutionProgress(
                            current_step=i + 1,
                            total_steps=total_steps,
                            step_name=step.tool_name,
                            status="failed",
                            message=f"✗ {step.tool_name} failed: {result.error}",
                            percentage=((i + 1) / total_steps) * 100
                        )
                        break
                        
                except Exception as e:
                    logger.error(f"Step {step.tool_name} failed: {e}")
                    step_end = datetime.now()
                    result = StepResult(
                        step=step,
                        success=False,
                        error=str(e),
                        execution_time=(step_end - step_start).total_seconds(),
                        end_time=step_end.isoformat()
                    )
                    step_results.append(result)
                    
                    yield ExecutionProgress(
                        current_step=i + 1,
                        total_steps=total_steps,
                        step_name=step.tool_name,
                        status="failed",
                        message=f"✗ {step.tool_name} failed: {str(e)}",
                        percentage=((i + 1) / total_steps) * 100
                    )
                    break
            
            # Final completion
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            success = all(result.success for result in step_results)
            if success:
                yield ExecutionProgress(
                    current_step=total_steps,
                    total_steps=total_steps,
                    step_name="pipeline",
                    status="completed",
                    message=f"🎉 Pipeline completed successfully in {total_time:.1f}s",
                    percentage=100.0
                )
            else:
                failed_steps = [r.step.tool_name for r in step_results if not r.success]
                yield ExecutionProgress(
                    current_step=total_steps,
                    total_steps=total_steps,
                    step_name="pipeline",
                    status="failed",
                    message=f"❌ Pipeline failed at steps: {', '.join(failed_steps)}",
                    percentage=100.0
                )
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            yield ExecutionProgress(
                current_step=0,
                total_steps=total_steps,
                step_name="pipeline",
                status="failed",
                message=f"❌ Pipeline execution failed: {str(e)}",
                percentage=0.0
            )
    
    async def _execute_step(self, step: ToolStep, input_mmif: str) -> StepResult:
        """Execute a single pipeline step."""
        logger.info(f"Executing step: {step.tool_name}")
        
        if step.tool_name not in self.tools:
            return StepResult(
                step=step,
                success=False,
                error=f"Tool {step.tool_name} not found"
            )
        
        tool = self.tools[step.tool_name]
        
        try:
            # Convert parameters to JSON string if needed
            parameters_json = json.dumps(step.parameters) if step.parameters else None
            
            # Execute the tool
            result = tool._run(
                input_mmif=input_mmif,
                config=step.config,
                parameters=parameters_json
            )
            
            # Check if result indicates an error
            try:
                result_data = json.loads(result)
                if isinstance(result_data, dict) and "error" in result_data:
                    return StepResult(
                        step=step,
                        success=False,
                        error=result_data["error"]
                    )
            except json.JSONDecodeError:
                # Result is not JSON, treat as successful MMIF output
                pass
            
            return StepResult(
                step=step,
                success=True,
                output=result
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return StepResult(
                step=step,
                success=False,
                error=str(e)
            )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def validate_plan(self, plan: PipelinePlan) -> List[str]:
        """Validate a pipeline plan and return any issues."""
        issues = []
        
        if not plan.steps:
            issues.append("Pipeline has no steps")
            return issues
        
        for i, step in enumerate(plan.steps):
            if step.tool_name not in self.tools:
                issues.append(f"Step {i+1}: Tool '{step.tool_name}' not available")
        
        return issues